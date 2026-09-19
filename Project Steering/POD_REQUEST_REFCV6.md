<!-- POD-REQUEST-REFCV6-DRAFT-2026-09-20 -->
# ⚠️ DRAFT — the refcv6 POD REQUEST, and the preparation that earns it

**Status: DRAFT.** It is landed while incomplete, on purpose: the PI parked the pod *"we will do it
later"* (2026-09-19), and a request that appears only when everything is finished is a request
nobody could check on the way. ⛔ **Every gap below is named as a gap, and the request is not made
until §5's list is empty.** Checklist item **E20** of `PREREG_REFCV6_DEVBOX_PREPARATION.md` is
discharged by §4 of this file.

---

## 1. What the dev box PROVED (each line has an artifact; none is a capability claim)

| # | proven | evidence |
|---|---|---|
| **P1** | The chain carries end to end at the ruled **416 × 1024**, every head live | §10.6 closure, 900 windows / 139 clips |
| **P2** | ⭐ `resnet101` — §10.2's PRIMARY trunk — **FITS** the 8 GiB card: **2.887 GB** @ b1, **3.890 GB** @ b2, all 22 heads live | `8b1f1db`; levers `--trunk-chunk-ckpt 1 --trunk-frozen-bn`, real stamped flags (`0f6036d`) |
| **P3** | The dev-box rate is **2.51 s/step** (197 marginal deltas, production logging) — the 28–29 s/step this plan was built on was **host paging**, not compute | `a5e6710` |
| **P4** | A **clean held-out set** exists: 124 clips, two disjoint 62/62 halves, parity-guarded, every clip mapped | `782571c`; 11 of the 139 eval clips were inside the parity TRAIN corpus (`c93182e`) |
| **P5** | The occupancy head clears its no-information floor **HELD OUT**: IoU **0.576** vs **0.3388** = **1.70×** (A3), confirmed at **0.584 = 1.72×** at 5,000 steps (A8) | `6e2ea33`, `b3feb4d` |
| **P6** | ⚠️ **More dev-box steps are NOT the lever for that head**: A8's curve is not monotone and its 2,000 → 5,000 gain (+0.0147) sits inside the rig's own run-to-run spread (0.007 at 2k, 0.040 at 1k, n = 2) | `b3feb4d` |
| **P7** | The refcv6 eval produces **all four metric families with paired episode-cluster intervals**, end to end, from a refcv6 checkpoint | W0–W3 (`5fd7c63`, `4000946`, `f7ad8b5`, `603bd32`) |
| **P8** | The conflict detector costs **+79.8 %** per step at 416 × 1024, with its three identity controls EXACT | `37645fc` |
| **P9** | The training source frames are **already on this box** (4,719/4,719 matching HF size + LFS sha256, 61.6 GB) ⇒ C3's transfer leg is void | `2a88524` |
| **P10** | HF has room: private **176.9 GB of 1 TB**; ⛔ the ceiling **BILLS** rather than refuses, so every push carries pre-push arithmetic | `2a88524`, ITEM 26 |

## 2. What the dev box CANNOT do, with the arithmetic

* **Sample-matched work.** This box runs `--batch 2` where a pod arm runs `--batch 20`, so matching
  the data a banked arm saw costs **10× the steps**: `resnet101` is **2.28 d step-matched** and
  **39.9–45.7 d sample-matched**; the 10-arm panel is **3.89 d at the 12,000-step cut** and
  **130.6 d sample-matched** (`aa3dfbb` §R.5).
* **Corpus scale.** The held-out set here is **124 clips**, 1/38th of the 4,713-clip corpus.
* **The driving claim.** `E-REFCV6V2-DRIVE` needs T1 on a corpus-scale held-out set with a training
  replicate. Both the compute and the corpus forbid it here; either alone would.

## 3. THE ASK

| what | ask | why this number |
|---|---|---|
| **GPU** | the PI's choice of **4 × 48 GB** or **2 × 80 GB** (dossier default: 4 × 48 GB) | D10 |
| **Volume** | ⭐ **≥ 600 GB** | the cache build peaks at **~458 GB** and sits at ~391 GB before checkpoints; the last measured `/workspace` quota was ~466 GB. ⛔ Verify with a real `dd` write test, never `df` |
| **First job** | a **200-step rate measurement**, before any arm | `SPEC` §11 R1: the token count rises 1.63× at the new geometry ⇒ *"re-measure rather than scale"*; the A40 rates on record (4.0–4.21 s/step) were all taken at 256 × 640 |
| **Data path** | ⭐ **build the cache ON the pod** (route c) | 76.2 GB down from the private HF corpus (~0.2–0.7 h) + 0.32 GB up, then **2–8 h of CPU build**. The alternatives are **86–92 h**: this box uploads at a MEASURED **1.2 MB/s**. Cache size re-derived: **380.6 GB** (370.3 train over the **4,572**-clip v8 split + 10.35 clean-124) |
| **Build host** | a **CPU pod on the same volume** for the build | the build needs no GPU; it keeps 2–8 h off the GPU bill |
| **Order** | **Tier 0 first**: rebuild the 139 eval episodes on the pod and match this box's per-episode sha256 | ⭐ the builder's POSITIVE CONTROL, before 4,572 clips depend on it |

## 4. ⛔ E20 — what the request must SAY about this rig's limits

1. **There is no `--resume`.** A dead arm **restarts from zero**, and `MAX_RELAUNCH=1` forbids the
   automatic relaunch. ⇒ the pod plan must budget checkpoint cadence and a done-marker
   (`summary.json`) per arm, and never assume a long arm can be resumed after a crash.
2. **There is no gradient accumulation.** The dev box cannot reach the pod's effective batch by
   configuration ⇒ **a dev-box arm and a pod arm are not the same experiment**, and every
   dev-box number quoted in support of a pod arm carries §1.2's step-vs-sample conversion with it.
3. **The memory levers change the arm.** `--trunk-frozen-bn` pins BN to stored statistics; there is
   **no configuration that both fits this card and reproduces an unfrozen-BN arm**. Any comparison
   against a banked unfrozen arm says so. On the pod this lever is not needed and should be OFF —
   which makes the pod arm the first unfrozen-BN arm of the line.
4. **Two build inputs do not exist yet** and are built ON the pod, after the cache: the train-side
   **3-D agent join** and the train-side **rig-extrinsics table**. Their run time is unmeasured.
5. **SAM3 gates training, not the build**: the cache build reads no map.

## 5. ⛔ NOT YET TRUE — the request is not made until this list is empty

* **E13 — A7**, the ImageNet knockout (2 conditions × 2 seeds). ⛔ **Blocked on the GPU**, which the
  PI's own reconstruction-studio servers hold; the panel is armed and launches when the card frees.
  It is the **only admissible lever claim** the dev box can buy.
* **E11/E12 — the S1 inference-only arms** (`S1-RANDOM` at the no-information value,
  `S1-GATE-CONST` at EXACTLY 0). The pass is running on CPU.
* **E9 — the T1 route's s/window** at 416 × 1024 (`refcv3_arm.py`; `t1_eval.py` does not run refc
  checkpoints by design).
* **`D-S1-DEP-BOX`** — the held-out BOX read (AP vs its base rate, velocity MAE) that
  `S1-GATE-PRED` actually depends on.
* **`D-DAC-HUMAN-ZERO-1`** — the drivable-compliance term zeroes the recorded human on **44.6 %** of
  held-out windows. Any pod plan that quotes PDMS/EP/DAC, and the D9 reward repair, are gated on
  its resolution.
* **PI decisions still open:** item 6 (the optional traffic-light spot-check) and v6F's fate.

---

⭐ **The sentence this request will be able to make, and cannot yet:** *here is the chain carrying
end to end at your geometry, every instrument reading a known value; here is the trunk that fits
and its measured rate; here is the held-out occupancy your collision gate was gated on; here is the
one lever claim an 8 GiB card could buy and what it cost; and here is the arithmetic showing the
remainder is not a dev-box question.*
