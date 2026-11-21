# refcv3 @ ckpt 40,284 — the four-family BASELINE, re-analysed with the labels fix

**Purpose.** The reference table refcv4b is to be compared against. Produced with **ZERO GPU**
(`--analyze-only` on the banked dump).

**Evidence class: MEASURED (ours).** Artifact: `REFCV3_BASELINE_REANALYZED.json` (this directory).
**Tier: T1** for `os` / `ha` / `ha0` / `os_navshuf` / `os_navzero`; **T0** for `oracle_sel`.
⚠️ The T1 stamp on the `os*` arms carries `tier_ruling: UNRULED` — see §6.

---

## 0. ⛔ THE HEADLINE CORRECTION — read this before quoting anything below

The task this work was commissioned under stated that the STRATEGIC nav-compliance family was
missing from every refcv3 eval because `corpus["labels"]` was a **dict** where the reader wanted a
path, and that the retroactive `resolve_labels_path` fix therefore makes banked dumps readable
**without a re-roll**. **The first half is true; the second half does not hold for these dumps.**

| claim | verdict | evidence |
|---|---|---|
| `corpus["labels"]` is a dict on the banked dumps | ⭐ **CONFIRMED** | both dumps: `type(corpus["labels"]) == dict`, `corpus["labels_path"] is None` |
| `resolve_labels_path` accepts that shape | ⭐ **CONFIRMED** | `taniteval/taniteval/nav_compliance.py:1013-1035` |
| the STRATEGIC family was missing from every refcv3 eval | ⛔ **REFUTED** | the record-level strategic block was **already present** in the OLD `refcv3-40284-openloop.ARM.json` with identical numbers (n=3622, acc 0.7667, kappa 0.4604 [0.3775, 0.5432]) |
| the fix recovers nav-compliance without a re-roll | ⛔ **REFUTED for these dumps** | see below |

**Why the fix cannot pay out here.** `from_refcv3_dump` checks for the ten required sidecar keys
**BEFORE** it ever calls `resolve_labels_path`:

```
taniteval/taniteval/nav_compliance.py:1046-1056
    need = ("plan_full_nav_true", "gstr_nav_true", "gt_future_ext", "pose_last",
            "ego_t0", "ep_poses", "nav_cmd", "nav_cmd_shuf", "nav_valid", "ws")
    missing = [k for k in need if k not in eps[0]]
    if missing:
        return unavailable_block(...)          # <-- returns HERE
    labels_path = resolve_labels_path(labels_path, ...)   # <-- never reached
```

**MEASURED, independently of the tool** (`probe_keys.py`, union over all 141 sidecar files):
six of the ten keys — `plan_full_nav_true`, `gstr_nav_true`, `gt_future_ext`, `pose_last`,
`ego_t0`, `ep_poses` — are **absent from all 141 files, in both npz sets**, in **both** banked
dumps (40,284 and 30k). The labels-dict defect is real and fixed; it is simply **not the binding
constraint** for these dumps. The binding constraint is that both were rolled before the sidecar
banked the nav-compliance geometry.

⇒ **What this re-analysis actually recovered is the `nav_compliance` KEY, not its NUMBERS.** It now
reports `status: UNAVAILABLE` with a reason and `n=4823`, which satisfies clause 5 of the
four-families rule (a family that cannot be computed states its reason and its n) instead of being
silently absent. **Trajectory-level nav compliance for refcv3 requires a GPU re-roll.**

---

## 1. Provenance

| item | value |
|---|---|
| dump | `taniteval/results/refcv3-40284-openloop-dump.tar.gz` |
| ckpt | `/workspace/experiments/refcv3-b1-v72-30k/ckpt_40284_FINAL.pt`, step **40284** |
| params | 107,032,901 total (core 104,879,522) — config.json vs rebuilt cross-check MATCHES |
| grid | `2s`: dt 0.5 s, 4 slots at 0.5 / 1.0 / 1.5 / 2.0 s; window stride 5, obs window 8 |
| **n windows** | **4,823** (0 skipped) |
| **n episodes** | **141** (141 available) |
| labels | `s2_labels_v7.2_eval.jsonl.gz`, md5 **`aa12c948f062181c3297265b51526ec5`** — matches the md5 the run recorded; 147 records, 141/141 episodes joined, 0 missing |
| lead block | `b1_eval_lead_block.npz` — PRESENT; LEAD 1489 / NO_LEAD 1145 / NOT_STRAIGHT 1889 / NO_LABEL 300; speed-check max 2.6e-05 m/s |
| estimator | **episode-cluster bootstrap** (`taniteval/ci.py`), n_boot 2000, seed 0; margins are the **paired** version |

**Exact commands** (CPU only, off-Drive clone — the G: mount cannot import the stack):

```bash
# clone (robocopy, 3 passes — one pass is not enough)
robocopy "<repo>\stack"     "C:\Users\Admin\tanitad-refcv4b-analyze\stack"     /E /MT:8
robocopy "<repo>\taniteval" "C:\Users\Admin\tanitad-refcv4b-analyze\taniteval" /E /MT:8

# extract  (⚠️ --force-local: MSYS tar reads "C:/..." as a remote host spec)
tar --force-local -xzf refcv3-40284-openloop-dump.tar.gz

export PYTHONIOENCODING=utf-8          # console is cp1252; banners carry non-ASCII
export PYTHONPATH="C:/Users/Admin/tanitad-refcv4b-analyze/stack;C:/Users/Admin/tanitad-refcv4b-analyze/taniteval"
export OMP_NUM_THREADS=6

C:/Users/Admin/venvs/tanitad/Scripts/python.exe taniteval/tools/refcv3_arm.py \
  --analyze-only <scratch>/dump40284/refcv3_40284_dump \
  --labels C:/Users/Admin/tanitad-refcv4b-analyze/labels/s2_labels_v7.2_eval.jsonl.gz \
  --n-boot 2000 --seed 0 --arm refcv3-40284-openloop \
  --out <scratch>/refcv3-40284-REANALYZED.json
```

⚠️ `--labels` is **required** even though the manifest carries a path: the recorded path is the
pod's (`/workspace/TanitAD/data/...`), which does not exist off-pod.

---

## 2. Arms

| arm | tier | meaning (abridged) |
|---|---|---|
| `os` | T1 * | **the deployed arm.** ONE forward at t0: observed frames + v7.2 nav token + MEASURED v0; path = the model's own `sel_score_v3` selection. No action input, no rollout. |
| `ha` | T1 | HOLD-ACTION control: the (a, steer) closing at t0, held. **Not a floor** — a held noisy steer can beat a straight line. |
| `ha0` | T1 | ⭐ **CONSTANT VELOCITY** (a=0, κ=0 at measured v0). The strongest trivial baseline and the reference floor. |
| `os_navshuf` | T1 * | `os` with nav permuted across windows — breaks pairing, preserves marginal. |
| `os_navzero` | T1 * | ⭐ `os` with nav WITHHELD (`nav_cmd=None`); E13 injection skipped entirely. **What deployment looks like.** |
| `oracle_sel` | **T0** | the `a_star` GT-nearest anchor — a ceiling, never comparable to T1. |
| `ol` | ABSENT | refcv3 consumes no actions (`refc_v3.py:480`), so an action-integrated arm is a property of the corpus, not of this model. |

\* tier stamped T1 with the ruling **OPEN**.

---

## 3. ADE / FDE per horizon — `n = 4,823 windows / 141 episodes` for every cell

Displacement error at each instant, m. **Estimator: episode-cluster bootstrap, 2000 draws, seed 0.**
Computed directly from the banked arrays; **CONTROL: the mean over the four slots reproduces the
record's own `intervals.ade_dense_m` to 4 dp on all six arms.**

| arm | 0.5 s | 1.0 s | 1.5 s | **2.0 s (FDE)** | **ADE** |
|---|---|---|---|---|---|
| `os` | 0.0730 [0.0672, 0.0801] | 0.2438 [0.2266, 0.2621] | 0.5221 [0.4850, 0.5607] | **0.9288** [0.8643, 0.9963] | **0.4419** [0.4114, 0.4734] |
| `ha` | 0.0518 [0.0457, 0.0599] | 0.1482 [0.1355, 0.1653] | 0.3397 [0.3119, 0.3721] | 0.6588 [0.6000, 0.7213] | 0.2996 [0.2749, 0.3280] |
| `ha0` | 0.1039 [0.0941, 0.1149] | 0.3732 [0.3355, 0.4101] | 0.8092 [0.7233, 0.8955] | 1.4029 [1.2517, 1.5595] | 0.6723 [0.6017, 0.7437] |
| `os_navshuf` | 0.0771 [0.0710, 0.0843] | 0.2515 [0.2341, 0.2693] | 0.5386 [0.5019, 0.5770] | 0.9582 [0.8921, 1.0266] | 0.4563 [0.4260, 0.4882] |
| `os_navzero` | 0.0904 [0.0806, 0.1016] | 0.2615 [0.2420, 0.2812] | 0.5460 [0.5070, 0.5862] | 0.9655 [0.8981, 1.0347] | 0.4659 [0.4337, 0.4987] |
| `oracle_sel` (T0) | 0.0671 [0.0618, 0.0741] | 0.2050 [0.1914, 0.2201] | 0.4180 [0.3903, 0.4477] | 0.7770 [0.7289, 0.8288] | 0.3668 [0.3440, 0.3909] |

⛔ **`ha` beats `os` on ADE by 0.1423 m (separated).** The deployed arm does **not** beat the
hold-action control. It beats `ha0` (constant velocity) by 0.2304 m. Both are reported in §5;
neither may be quoted alone.

---

## 4. The four families — per arm, never pooled

### 4.1 LONGITUDINAL — `n = 4,823 windows / 141 episodes`

Columns with [CI] are episode-cluster bootstrap (mean + 95 %); columns without are `full_set`
pooled point estimates.

| arm | speed MAE (m/s) | speed bias | speed RMSE | along MAE (m) | along bias | accel MAE (m/s²) |
|---|---|---|---|---|---|---|
| `os` | 0.4516 [0.4197, 0.4828] | +0.0327 | 0.7384 | 0.4030 [0.3722, 0.4332] | +0.0372 | 0.6806 |
| `ha` | 0.2540 [0.2364, 0.2718] | −0.0632 | 0.4480 | 0.2348 [0.2163, 0.2580] | −0.0777 | 0.3166 |
| `ha0` | 0.4880 [0.4415, 0.5332] | −0.0410 | 0.8093 | 0.4705 [0.4271, 0.5126] | −0.0262 | 0.4786 |
| `os_navshuf` | 0.4687 [0.4370, 0.5000] | +0.0024 | 0.7626 | 0.4182 [0.3883, 0.4485] | +0.0096 | 0.7041 |
| `os_navzero` | 0.4745 [0.4422, 0.5078] | −0.0421 | 0.7607 | 0.4276 [0.3950, 0.4615] | −0.0510 | 0.7012 |
| `oracle_sel` (T0) | 0.3929 [0.3696, 0.4164] | +0.0320 | 0.7008 | 0.3264 [0.3056, 0.3493] | +0.0229 | 0.7206 |

**Target-speed accuracy** (fraction of the 19,292 horizon STEPS within band):

| arm | ±0.5 m/s | ±1.0 m/s | ±2.0 m/s |
|---|---|---|---|
| `os` | 0.7135 | 0.8784 | 0.9713 |
| `ha` | 0.8662 | 0.9560 | 0.9932 |
| `ha0` | 0.7047 | 0.8569 | 0.9595 |
| `os_navshuf` | 0.7026 | 0.8732 | 0.9676 |
| `os_navzero` | 0.6922 | 0.8738 | 0.9678 |
| `oracle_sel` (T0) | 0.7596 | 0.9124 | 0.9823 |

**Distance-keeping: `status = OK`** on every arm (n = 1,235–1,315 lead-bearing windows of 4,823),
because the B1 eval lead block was supplied. Without `--lead-block` it REFUSES — that is a work
item, not a pass.

### 4.2 LATERAL — `n_windows = 4,823`; step counts as noted

| arm | heading MAE (deg) | yaw-rate MAE (deg/s) | curvature MAE (1/m) | curv bias | cross MAE (m) | cross final (m) | n steps head / curv |
|---|---|---|---|---|---|---|---|
| `os` | 1.3591 [0.8220, 2.3840] | 1.7940 | 0.008815 | −0.000200 | 0.1084 [0.0958, 0.1222] | 0.2337 | 18115 / 13538 |
| `ha` | 1.6430 [0.9251, 2.9788] | 1.4542 | 0.004030 | +0.000255 | 0.1226 [0.1040, 0.1419] | 0.2926 | 18082 / 13500 |
| `ha0` | 2.8715 [1.9883, 4.3359] | 2.3714 | 0.006802 | +0.001672 | 0.3132 [0.2499, 0.3879] | 0.6691 | 18077 / 13518 |
| `os_navshuf` | 1.4052 [0.8449, 2.4660] | 1.8154 | 0.008762 | +0.000272 | 0.1085 [0.0960, 0.1220] | 0.2332 | 18099 / 13523 |
| `os_navzero` | 1.4604 [0.8455, 2.6300] | 1.8793 | 0.009725 | +0.001177 | 0.1098 [0.0970, 0.1239] | 0.2355 | 18098 / 13516 |
| `oracle_sel` (T0) | 1.4596 [0.8562, 2.5385] | 1.9628 | 0.009919 | −0.000075 | 0.1033 [0.0916, 0.1159] | 0.2184 | 18125 / 13542 |

⚠️ **Reducer caveat.** The bootstrap mean and the `full_set` pooled point differ for heading:
`os` reads **1.3109** pooled vs **1.3591** bootstrap-mean. The interval is the decision-grade
object; quote the estimator with the number.
⚠️ 1,177 steps excluded below `min_ds = 0.25 m` (heading is undefined at rest).
⚠️ `ha0` is **constant-velocity on 100 % of windows** — read its LATERAL row as a control, never as
planning skill.

### 4.3 TACTICAL (trajectory-derived) — `n = 4,823 / 141`

Labeller: `tanitad.refs.refc_tactical.factor_from_kinematics`. **Read LAT and LON before the
collapsed 5-way** — the mixed softmax is the programme's named defect.

| arm | LAT acc | LAT κ [95 %] | LON acc | LON κ [95 %] | 5-way acc | 5-way κ [95 %] |
|---|---|---|---|---|---|---|
| `os` | 0.9540 | **0.8113** [0.7541, 0.8578] | 0.7477 | **0.3078** [0.2484, 0.3659] | 0.7578 | 0.5167 [0.4600, 0.5698] |
| `ha` | 0.9382 | 0.7374 [0.6863, 0.7824] | 0.8443 | 0.6071 [0.5672, 0.6441] | 0.8275 | 0.6664 [0.6324, 0.6993] |
| `ha0` | 0.8659 | **0.0000** [0.0000, 0.0000] | 0.7576 | **0.0000** [0.0000, 0.0000] | 0.6718 | **0.0000** [0.0000, 0.0000] |
| `os_navshuf` | 0.9560 | 0.8177 [0.7614, 0.8621] | 0.7396 | 0.2945 [0.2393, 0.3484] | 0.7512 | 0.5046 [0.4464, 0.5578] |
| `os_navzero` | 0.9533 | 0.8068 [0.7511, 0.8529] | 0.7468 | 0.3187 [0.2656, 0.3707] | 0.7562 | 0.5187 [0.4623, 0.5710] |
| `oracle_sel` (T0) | 0.9515 | 0.8032 [0.7428, 0.8497] | 0.7584 | 0.3940 [0.3373, 0.4444] | 0.7607 | 0.5478 [0.4937, 0.5953] |

⭐ **`ha0`'s κ = 0.0000 with CI [0, 0] is a STRUCTURAL ZERO, not a noisy estimate** — a
constant-velocity arm emits one class, so κ is identically zero. It is the control that must read a
known value, and it does.

⛔ **The longitudinal decision is the weak axis**: `os` κ = 0.3078 against a lateral κ of 0.8113 —
and the hold-action control **beats the model** on LON (0.6071). This is the programme's known
longitudinal-blindness signature, still present at 40,284.

**TACTICAL turn recall per class** (`os`, lateral_decision, n = 4,823):

| class | n_true | n_pred | **recall** | precision |
|---|---|---|---|---|
| `lane_keep` | 4,176 | 4,157 | **0.9715** | 0.9759 |
| `turn_left` | 251 | 239 | **0.8048** | 0.8452 |
| `turn_right` | 396 | 427 | **0.8636** | 0.8009 |

**Anchor selection** (the TACTICAL goal-setting half), n = 4,823 / 141, 128 anchors, chance 0.0078:

| quantity | value [95 %] |
|---|---|
| `anchor_acc` (argmax == a_star) | **0.5654** [0.5320, 0.5992] |
| `deployed_selection_agrees_oracle` | 0.5652 [0.5333, 0.5990] |

`goal_setting` reports `accuracy = None` — the sub-block exists but its scalar is unpopulated.

### 4.4 STRATEGIC

**Two distinct surfaces. Both must be stated; they are not interchangeable.**

**(a) Record-level route head — `refcv3.strategic`: PRESENT, n = 3,622 windows / 128 episodes**
(1,201 of 4,823 excluded for no route label). `nav_valid_frac = 1.0`.

| conditioning | accuracy [95 %] | **kappa [95 %]** | nav echo index | majority-class rate |
|---|---|---|---|---|
| `nav_true` | 0.7667 [0.7097, 0.8224] | **0.4604** [0.3775, 0.5432] | 0.1651 | 0.6742 |
| `nav_shuffled` | 0.7667 [0.7097, 0.8224] | 0.4604 [0.3775, 0.5432] | 0.1651 | 0.6742 |
| `nav_zero` | 0.7667 [0.7097, 0.8224] | 0.4604 [0.3775, 0.5432] | 0.1651 | 0.6742 |

**Route recall per class** (identical under all three conditionings):

| class | n_true | n_pred | **recall** | precision |
|---|---|---|---|---|
| `route_left` | 470 | 377 | **0.3681** | 0.4589 |
| `route_straight` | 2,442 | 2,867 | **0.9541** | 0.8127 |
| `route_right` | 710 | 378 | **0.3859** | 0.7249 |

⛔⛔ **NEW MEASURED FINDING — THE ROUTE HEAD IS NAV-INVARIANT.** The three conditionings are
identical because the banked route predictions are **bit-identical**:

| head | differs true vs shuffled | differs true vs zero | n |
|---|---|---|---|
| **`route`** | **0 / 4,823** | **0 / 4,823** | 4,823 |
| `lat` | 363 / 4,823 (7.53 %) | 754 / 4,823 (15.63 %) | 4,823 |
| `lon` | 550 / 4,823 (11.40 %) | 697 / 4,823 (14.45 %) | 4,823 |

⭐ **This is not an instrument artifact, and the same-breath control proves it.** The nav shuffle
genuinely changed the token on **2,406 / 4,823** windows (49.89 %), and `nav_injected_true = 1.0`
vs `nav_injected_zero = 0.0` on every window, so the intervention landed. The **lat and lon heads,
written by the same dump code in the same batch, DO move** — so the conditionings are distinct
forwards. The route head alone does not respond.

Corroborating, on the CHANGED subset (n = 1,736 windows / 127 episodes):

| quantity | value [95 %] |
|---|---|
| `route_follows_LABEL_under_shuffle` | **0.7362** [0.6729, 0.8001] |
| `route_follows_SHUFFLED_NAV_under_shuffle` | **0.2344** [0.2001, 0.2720] |
| `paired_true_minus_shuffled_accuracy` | **0.0000** [0.0000, 0.0000], separated = **False** |

⇒ refcv3's route head infers route from **vision**, not from the nav token. The `_echo_caveat` does
not bite: κ = 0.4604 is **not** a nav echo, and it is the **deployment-relevant** number because it
is unchanged when nav is withheld. ⚠️ Conversely, **the strategic head is not nav-controllable** —
a route command cannot steer it.

**(b) Per-arm trajectory strategic — `arms[*].four_families.strategic`: UNAVAILABLE, n = 0.**
Reason: `missing ['route_pred', 'route_gt']`; needs map-derived `optionset`
(`stack/experiments/nurec-gsplat/strategic_gt.py`). ⛔ Blocked on a corpus that does not exist:
PhysicalAI-AV ships **no map, lane graph or junction annotation**.

**(c) `refcv3.strategic.nav_compliance`: UNAVAILABLE, n = 4,823** — the key now exists (it did not
before); the numbers need a GPU re-roll. See §0.

---

## 5. Paired margins — paired episode-cluster bootstrap, n = 4,823 windows / 141 episodes

| comparison | metric | delta | 95 % CI | separated | p(Δ>0) |
|---|---|---|---|---|---|
| **`os` − `ha0`** | ADE (m) | **−0.2304** | [−0.2881, −0.1781] | **True** | 0.000 |
| **`os` − `ha0`** | speed MAE | −0.0364 | [−0.0636, −0.0079] | **True** | 0.006 |
| `os` − `ha` | ADE (m) | **+0.1423** | [+0.1187, +0.1658] | **True** | 1.000 |
| `os` − `ha` | speed MAE | +0.1976 | [+0.1737, +0.2204] | **True** | 1.000 |
| `os` − `os_navshuf` | ADE (m) | −0.0144 | [−0.0220, −0.0069] | **True** | 0.000 |
| `os` − `os_navshuf` | speed MAE | −0.0171 | [−0.0259, −0.0084] | **True** | 0.000 |
| **`os_navzero` − `ha0`** | ADE (m) | **−0.2064** | [−0.2673, −0.1479] | **True** | 0.000 |
| **`os_navzero` − `ha0`** | speed MAE | −0.0134 | [−0.0437, +0.0179] | **False** | 0.202 |
| `os` − `os_navzero` | ADE (m) | −0.0239 | [−0.0428, −0.0089] | **True** | 0.001 |
| `os` − `os_navzero` | speed MAE | −0.0229 | [−0.0378, −0.0102] | **True** | 0.000 |
| **`ha` − `ha0`** | ADE (m) | **−0.3727** | [−0.4346, −0.3161] | **True** | 0.000 |
| **`ha` − `ha0`** | speed MAE | −0.2339 | [−0.2699, −0.1967] | **True** | 0.000 |
| `oracle_sel` − `os` (T0−T1) | ADE (m) | −0.0751 | [−0.0884, −0.0618] | **True** | 0.000 |
| `oracle_sel` − `os` (T0−T1) | speed MAE | −0.0587 | [−0.0733, −0.0441] | **True** | 0.000 |

Negative = the first arm is better. **Only `ade_m` and `speed_mae_mps` carry paired margins**; the
LATERAL, TACTICAL and STRATEGIC families have per-arm intervals but **no paired margin** in this
record — a gap to close before any lever claim on those axes.

⛔⛔ **A separated CI here is NECESSARY, NOT SUFFICIENT.** This is a **one-seed** arm and the
episode-cluster bootstrap resamples EPISODES with the model held fixed — it answers *"would another
draw of episodes say this?"*, never *"would another training run say this?"* (`H-ESTIM-SEED-1`;
a replicate arm showed a ~17 % false-positive rate for `separated` on the v7-tiny rig). The
`os − os_navshuf` margin of **−0.0144 m** and the `os − os_navzero` margin of **−0.0239 m** are
exactly the size at which that matters. **Do not read them as lever effects without a replicate.**

---

## 6. Shape controls — read before any family row

**Trivial profile** (n = 4,823; straight |y| < 1e-6 m, const-speed spread < 1e-4 m):

| arm | straight_frac | const_speed_frac | trivial (const-velocity) frac |
|---|---|---|---|
| `os` | 0.0000 | 0.0000 | 0.0000 |
| `ha` | 0.0402 | 0.0319 | 0.0311 |
| **`ha0`** | **1.0000** | **1.0000** | **1.0000** |
| `os_navshuf` | 0.0000 | 0.0000 | 0.0000 |
| `os_navzero` | 0.0000 | 0.0000 | 0.0000 |
| `oracle_sel` | 0.0000 | 0.0000 | 0.0000 |

⛔ **VOID-RISK, reported by the instrument itself:**
- `os` is **bit-identical to `oracle_sel` on 2,726 / 4,823 windows (56.52 %)**
- `os` is **bit-identical to `os_navshuf` on 2,417 / 4,823 windows (50.11 %)**

On those windows the instrument saw ONE arm, not two. The `os − os_navshuf` margin is measured on
the ~50 % of windows where the two arms differ at all.

**Selection profile** — the model's own choice over 128 anchors:

| quantity | value |
|---|---|
| **`n_distinct_selected`** | **50** of 128 |
| **modal anchor** | **#57, share 0.1482** |
| **selection entropy** | **2.8425 nats** of max 4.8520 (**ratio 0.5858**) |
| `agrees_with_oracle_frac` | 0.5652 |
| `degenerate` | **False** (threshold: modal_frac ≥ 0.90) |

Not degenerate — the model is not emitting one anchor for every scene.

⚠️ **Tier ruling OPEN.** `EVAL_DOCTRINE`'s T1 row reads *"the predictor consumes the
decoder/planner's own actions"*, which does not literally cover a model consuming **no** actions.
Benchmarks recommends ADMIT with the distinct arm name `os`. Every headline here is a margin over
`ha0` on the same windows with the same instrument, so the ruling changes the **label**, never the
arithmetic.

---

## 7. Blockers — what would be needed to complete the STRATEGIC family

| item | blocked on | what would unblock it |
|---|---|---|
| `nav_compliance` (trajectory-level nav compliance) | **GPU** | re-roll both dumps with the current `refcv3_arm.py`, which banks `plan_full_nav_true`, `gstr_nav_true`, `gt_future_ext`, `pose_last`, `ego_t0`, `ep_poses`. Nothing is computable post hoc. |
| per-arm `four_families.strategic` | **a corpus that does not exist** | map-derived `optionset`; PhysicalAI-AV ships no map / lane graph / junction annotation. Would need AlpaSim or NuRec (`map.xodr`). |
| paired margins on LATERAL / TACTICAL / STRATEGIC | **instrument** | `families_paired` currently carries only `ade_m` and `speed_mae_mps`. Zero-GPU work item. |
| any lever claim from a `separated` flag here | **a replicate arm** | one training/inference seed only — see `H-ESTIM-SEED-1`. |

---

## 8. Defects found (file:line)

1. ⭐ **`taniteval/taniteval/nav_compliance.py:1046-1056`** — the required-keys guard returns before
   `resolve_labels_path` (line 1013) is reached. **Not a bug**; it is correct ordering. But it means
   the labels-dict fix cannot recover nav-compliance from any dump that predates the sidecar keys,
   which is **both** banked refcv3 dumps. The commissioning premise that banked dumps are readable
   without a re-roll does not hold here.
2. ⚠️ **`taniteval/tools/refcv3_arm.py:1913`** writes `corpus["labels_path"] = a.labels`, but the
   `**join` splat that follows still overwrites `corpus["labels"]` with the provenance dict. Both
   dumps read `labels_path: None`. Harmless now that `resolve_labels_path` accepts all three shapes,
   but the new key is **not** populated in these dumps.
3. ⚠️ **MEASURED, not a code defect** — the route head is bit-identical across all three nav
   conditionings (§4.4). Any strategic claim conditioned on nav for refcv3 is measuring vision.

---

## 9. Files

| file | location |
|---|---|
| re-analysed record | `REFCV3_BASELINE_REANALYZED.json` (this directory) |
| this table | `REFCV3_BASELINE_REANALYZED.md` (this directory) |
| source dump | `taniteval/results/refcv3-40284-openloop-dump.tar.gz` (repo) |
| prior record (no `nav_compliance` key) | `taniteval/results/refcv3-40284-openloop.ARM.json` (repo) |
| labels blob | `s2_labels_v7.2_eval.jsonl.gz`, md5 `aa12c948f062181c3297265b51526ec5` |
| off-Drive clone | `C:\Users\Admin\tanitad-refcv4b-analyze` (working dir, not banked) |

**Control readings for the STRATEGIC recovery claim:**

| probe | reading | verdict |
|---|---|---|
| `grep -cF "nav_compliance"` on the NEW json | **1** | key present |
| `grep -cF "nav_compliance"` on the OLD `.ARM.json` | **0** | genuinely absent |
| `grep -cF "refcv3"` on the OLD `.ARM.json` (readability control) | **38** (628,198 bytes) | the 0 above is a real absence, not an unreadable file |
| `refcv3.strategic.nav_compliance.status` | **UNAVAILABLE** | ⛔ the key was recovered; the numbers were not |
| `refcv3.strategic.conditionings.nav_true.status` | **OK, n = 3,622** | the strategic route family IS present — and was present before this run too |
