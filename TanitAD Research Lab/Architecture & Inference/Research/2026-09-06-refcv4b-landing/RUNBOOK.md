# RUNBOOK — the refcv4b landing sequence (every step PREFLIGHTED on `ckpt_30000.pt`)

⭐ **Nothing below is a first attempt.** Every command in §2–§4 has already been run end to end on
the pod against `ckpt_30000.pt`, CPU-only, exit 0, with its output inspected. The landing changes
**one argument** — the checkpoint — plus the scale knobs named per step.

---

## 0. GATE 0 — `pod:/workspace/land_gate0.sh` (md5 `29dbb0303ae0502700b14b92a2cf3718`)

```bash
ssh -n tanitad-refcv3 'bash /workspace/land_gate0.sh'
```

Emits opaque `GG…GG` markers and **refuses** unless all five hold:

1. `summary.json` parses and `done is True` (written by the supervisor itself);
2. `metrics.jsonl`'s own max step ≥ 40,284 — an assertion **independent of the marker**;
3. `sup_refcv4b` **and** `refc_v3_train` process counts both read **0** (character-class greps, so
   no grep matches its own argv). If either is alive it prints the PIDs and **stops** — kill by
   **explicit PID**, never `pkill -f`;
4. the rolling `ckpt.pt` is copied to the immutable `ckpt_40284_FINAL.pt` and **both md5s are
   32 chars and equal** — a length-guarded comparison, because two empty strings compare equal
   and that has reported a phantom success before;
5. the checkpoint's **own** `step` field reads 40,284, with `len(state_dict) > 400` as a control.

⚠️ **Why the freeze matters here specifically:** `MILESTONES = (5000, 15000, 20000, 30000)` does
**not** contain 40,284, so the final weights live in the **rolling** `ckpt.pt` — a file whose name
never changes while its contents do (`refc_v3_train.py:1536`, `step % save_every == 0 or step ==
args.steps`). A reel or an eval rendered from a stale copy is indistinguishable from the real one
except by this check.

**DRY-RUN VERIFIED 2026-09-05T23:3xZ:** the script executes and correctly returns
`GGGATE0=FAILGG no summary.json` (RC 1) while the run is live — its failure path is proven, not assumed.

---

## 0b. ⛔ THE `stack/` SYNC HAPPENS AFTER GATE 0, NEVER BEFORE

A full md5 manifest of `stack/tanitad/**/*.py` (repo 164 files, pod 133) classifies as
**118 identical (the control, non-zero) · 15 different · 31 missing on the pod · 0 pod-only**.
Four of the drifted files are on the eval path's module-level imports (`data/comma2k19.py`,
`models/agent_slots.py`, `models/metric_dynamics.py`, `models/v6.py`).

⛔ **It must NOT be shipped while the run is live.** The registry's own Location note says
*"never ship a trainer file to the pod mid-run"*, and the mechanism is concrete: the supervisor
relaunches the trainer on death, and a relaunch would import whatever is on disk — i.e. a code
change could alter the model **mid-run**. MEASURED: the trainer is a single process,
PID 2346318, `lstart` **Fri Sep 4 11:40:23 2026** — it has never relaunched, so nothing has
happened; that is luck to preserve, not a licence.

⚠️ The **taniteval** sync (§3.1 of `STATUS.md`) was safe and was checked, not assumed:
`refc_v3_train.py` imports `taniteval` **0 times** (control: 29 import lines in the file, so the
probe reads), and the trainer has not relaunched since two days before the sync.

**Post-GATE-0 sequence:** ship `raw/stage_stack.sh`'s verified set (44 files, ⛔ **excluding
`refs/refc.py` and `refs/refc_v3.py`** — the code that trained), then **re-run the 20-clip BASE
probe and compare it to `PRELIM_CPU_PANEL.md` §1**. If the numbers move, the preliminary panel was
produced under stale code and that is reported, not quietly dropped.

---

## 1. What the eval will run on, and the one decision that is not cost-driven

| | |
|---|---|
| adapter | `taniteval/tools/refcv3_arm.py` (its docstring `:447` binds it to *"every REF-C v3/v4 build — including refcv4b"*) |
| model code | ⛔ the **POD's** `stack/tanitad/refs/refc.py` + `refc_v3.py` — the code that trained. The repo's `refc.py` is +589 lines of a sibling's uncommitted refcv5 wiring. |
| episodes | `/root/data/eval` — **141** clips, held out (`refc_v3_train.py:1295-1301` REFUSES any train∩eval overlap) |
| labels | `/workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz`, md5 `aa12c948f062181c3297265b51526ec5` — the same blob the run's `config.json` records |
| **`--window-stride 5`** | ⭐ **chosen for COMPARABILITY, not for cost.** refcv3 @40,284 was scored at stride 5 → **4,823 windows / 141 episodes**. Same stride ⇒ same window grid ⇒ the refcv3 baseline is a like-for-like comparison instead of two different experiments. |
| estimator | paired episode-cluster bootstrap, `--n-boot 2000 --seed 0` |

### The arms, and what each one is for

| arm | tier | why it must be there |
|---|---|---|
| `os` | T1 (ruling UNRULED) | the deployed one-shot planner: `out["traj"]`, the model's own `sel_score_v3` pick — ⛔ never `a_star` |
| `ha` | T1 | hold-action. ⚠️ **`echo_gate.py:119-132` reads it at source and concludes "`ha` IS `ha0_ext`"** — the weaker form (accel by finite difference, steer at t0−1) |
| `ha0` | T1 | constant velocity at the measured v0 — the strongest trivial baseline and the only arm bit-comparable with refav1's |
| **`ha0_ext`** | T1 | ⭐ **THE ECHO CONTROL** — constant a0 **and** constant κ0, both at t0. `refcv3_arm.py:1399` already produces it; the doc claiming otherwise is stale |
| `os_navshuf` | T1 | breaks the nav PAIRING, preserves the marginal |
| `os_navzero` | T1 | removes the nav SIGNAL — **the deployment-relevant arm**, since nav is an oracle |
| `oracle_sel` | **T0** | the ceiling. ⛔ never compared to a T1 number as a level |
| `ol` | **ABSENT** | refcv3/v4 consume no recorded actions, so integrating them is a corpus property, not a rollout of this model — written into the record **with its reason** (`refcv3_arm.py:26`), never dropped. ⛔ `oracle_s0` is **not an arm anywhere in the code**; it is a refav1 *run-directory name*. |

---

## 2. Step 1 — the cost probe (2 clips, GPU), which SETS nothing but the wall-clock estimate

```bash
ssh -n tanitad-refcv3 'cd /workspace/TanitAD && OMP_NUM_THREADS=6 \
 PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval PYTHONIOENCODING=utf-8 \
 python3 taniteval/tools/refcv3_arm.py \
  --ckpt /workspace/experiments/refcv4b-b1-v72-40k/ckpt_40284_FINAL.pt \
  --config /workspace/experiments/refcv4b-b1-v72-40k/config.json \
  --episodes /root/data/eval \
  --labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --nav-source v72 --grid 2s --action-units steer --device cuda \
  --episodes-n 2 --window-stride 5 --with-oracle-sel \
  --lead-block /workspace/eval/b1_eval_lead_block.npz \
  --n-boot 200 --seed 0 \
  --dump-dir /workspace/eval/refcv4b_probe_dump --out /workspace/eval/refcv4b_probe.json \
  --tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0'
```

Read the `[cost]` line. ⛔ Quote that, never a wall-clock from this file.

## 3. Step 2 — the real read (all 141 clips)

Identical, with `--episodes-n` removed, `--n-boot 2000`, and
`--dump-dir /workspace/eval/refcv4b_t1_dump --out /workspace/eval/refcv4b_t1.json`.

⛔ **`--analyze-only /workspace/eval/refcv4b_t1_dump` before re-running anything.** The rollout is
the only expensive part; an analysis-time failure is recoverable with zero GPU. *(That trap has
already destroyed one paid-for 2-arm / 40-episode rollout in this programme.)*

### What must be read, in this order — and a degenerate profile makes the read VOID, not negative

1. `[model]` / `[grid]` / `[units]` provenance;
2. the **trivial profile** — per arm `straight_frac`, `const_speed_frac`, `CONSTANT-VELOCITY`,
   `identical_to`. ⚠️ In the 1-clip preflight `os` was bit-identical to `os_navshuf` on 5/5 — an
   artifact of shuffling within ONE clip (`nav_shuffle changed 0/5`); at 141 clips the shuffle
   changes the token on roughly half the windows (2,406/4,823 on the refcv3 surface);
3. the **selection profile** — `n_distinct_selected`, modal anchor + share, selection entropy,
   agreement with the oracle;
4. only then the four families and the paired margins, **including `os_navzero − ha0` beside
   `os − ha0`**, and **`os − ha0_ext`** (the echo bar).

### Then the four checks that are FAILURES if absent, not omissions

```bash
# (a) STRATEGIC present and non-empty, with a control that must read non-zero
ssh -n tanitad-refcv3 'cd /workspace/eval && \
  echo "navcomp=$(grep -c nav_compliance refcv4b_t1.json)" && \
  echo "CONTROL_ha0ext=$(grep -c ha0_ext refcv4b_t1.json)"'

# (b) the EVAL-SET kin3 marginal -- closes the scope gap in the "worse than its
#     own prior" finding (the buffer in the ckpt is a TRAIN EMA)
ssh -n tanitad-refcv3 'cd /workspace/TanitAD && OMP_NUM_THREADS=4 \
 PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval PYTHONIOENCODING=utf-8 \
 python3 /workspace/eval/kin3_marginal.py /workspace/eval/refcv4b_t1_dump \
   /workspace/eval/refcv4b_kin3_marginal.json'

# (c) turn recall PER CLASS beside ADE  -- four_families.tactical.lateral_decision.per_class
# (d) TanitAD_BenchmarkCriteria over refcv4b_t1.json; every gap is a WORK ITEM
```

⭐ **(b) is already proven end to end** on the preflight dump (n=5): it recomputes the labels with
the trainer's own `refc_tactical.window_factored_labels(pose_last, gt_future_ext[:, :20])`
(`refc_v3_train.py:578`) from two fields the decisions sidecar already carries, and asserts both
marginals sum to 1 and the counts sum to n. ⚠️ The sidecar's own `lat_label`/`lon_label` are the
**8-wide v7.2** labels with −100 = IGNORE (verified: values reach 7 and −100), **not** kin3 — the
core's aux labels are never dumped, which is why they must be recomputed.

---

## 4. The reel — 15 clips, 2,571 frames, **257.1 s = 4.95× the refcv3 reel**

### 4.0 The calibration constraint, and why it turned out not to bind

The camera panel needs **MEASURED per-clip extrinsics** and PhysicalAI-AV is gated, so the first
plan was bounded by the two banked extrinsics files (the refcv3 reel's 3 clips + the refav1 reel's
8 = 9 distinct, all 9 present in the eval cache). ⭐ **That constraint was false:** the calibration
source is on the dev box at `C:/Users/Admin/tanitad-data/physicalai/` (`clip_index.parquet` +
`calibration/sensor_extrinsics`), so `pai_extrinsics_table.py` builds the table for **all 141
eval clips** — `raw/extrinsics141.json`, shipped to `pod:/workspace/eval/extrinsics141.json`
(md5 `795d09cd32f52c74f6ab2ca2739e9a3b`).

**CONTROL, and it is what makes the new table trustworthy:** on the **9** clips the two banked
files also cover, the newly built table reproduces every one of `x, y, z, qx, qy, qz, qw` to
< 1e-6 — **0 field mismatches over 9 shared clips** (the count of compared clips is printed, so a
vacuous "0 mismatches" from an empty intersection is excluded). Camera height over the 141 spans
**1.213–1.662 m, median 1.316** — consistent with the 1.245–1.607/1.306 measured over a different
40-clip set, and it re-confirms that the three constants circulating in this repo (1.22, 1.43, 1.5)
are all wrong as a constant; 1.22 is below the observed minimum.

### 4.1 ⭐ THE CLIP SELECTION IS A RULE, NOT A PICK

The banked refcv3 reel carries the caveat *"this is a hand-picked reel and it must never be quoted
as a representative one."* The only way to drop that caveat is a criterion that is **measured,
stated in advance, and computed from ego kinematics alone — never from how the model does.**

> **THE RULE:** from the 141 held-out eval clips, take the **top 8 by `abs_turn_deg`** (total
> absolute yaw turned over the clip) **∪** the **top 8 by `v_span`** (speed range, `v_max − v_min`).

The two lists share one clip (`ed87040c`), so the union is **15 clips**. Computed by
`raw/clip_profile.py`, which reads **only poses** out of each `*.v2ep.pt` — no frames, no model, no
GPU — and is banked as `raw/clip_profile.json`.

| | |
|---|---|
| clips | **15** |
| windows at `--stride 1` | **2,571** (`n_frames − (n_stack−1) − window − max_horizon`) |
| duration @ 10 fps | **257.1 s = 4 min 17 s** |
| vs the banked refcv3 reel | **4.95×** (519 frames / 51.90 s) |
| vs the refav1 `dense8` reel | **1.64×** (1,564 frames / 156.40 s) |
| lateral coverage | **5** clips net-left (> +15°), **6** net-right (< −15°); max ±199.8° turned |
| longitudinal coverage | **7** clips reach a full stop (`v_min` < 1 m/s), **6** exceed 15 m/s; max `v_span` 16.2 m/s |

⇒ both lateral directions and both longitudinal signs are exercised by construction, which is what
makes the reel readable against the four binding families rather than a highlight tape.

```
--clips "6ed4ef7a-1a70-48c5-8dc1-3c6a48ec3ecc,63434e0b-8cd2-4e1f-9526-8a06bd7f4b0d,\
73e750eb-684f-4aa1-953e-eb0f671a86b7,9c50f803-c78c-4b69-8a83-443ad312553e,\
ed87040c-cf3c-4fe4-9a95-50c1114f0d38,77a247be-ad56-4712-8e16-5d4acc5c241d,\
3ef3dcc8-7fd8-44df-837e-2820e969e691,ca11a2a2-1021-475c-9907-e6dbe658ad12,\
24fee8a5-9ccc-4610-b27b-560f08fd46c8,14bc9fcf-c048-42c3-b35d-0fc13b8ddc04,\
a0d95df6-499b-45ee-a424-5954f6e65f77,91e90a95-a129-486a-a1d4-cc1a2915793f,\
4c5264dd-9110-46d4-88e7-693ffde925e5,330b4626-96d0-46fd-a10b-3532f389bd34,\
c8a39711-ce08-421f-b675-6533f89ad8a3"
```

⚠️ The reel is a **stress set**, not a random sample — it over-represents turning and speed change
on purpose, and the per-clip ADE in the sidecar must never be averaged into a headline. The
representative read is the 4,823-window four-family panel in §3.

```bash
ssh -n tanitad-refcv3 'cd /workspace/TanitAD && OMP_NUM_THREADS=6 \
 PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval PYTHONIOENCODING=utf-8 \
 python3 taniteval/tools/render_refcv3_video.py \
  --ckpt /workspace/experiments/refcv4b-b1-v72-40k/ckpt_40284_FINAL.pt \
  --config /workspace/experiments/refcv4b-b1-v72-40k/config.json \
  --episodes /root/data/eval \
  --labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --extrinsics /workspace/eval/extrinsics141.json \
  --nav-source v72 --grid 2s --action-units steer --device cuda \
  --stride 1 --fps 10 --with-oracle --expect-step 40284 --keep-frames \
  --clips "d85682b8-...,ca11a2a2-...,6ed4ef7a-...,142a3a72-...,1c3a2c7c-...,24fee8a5-...,c8a39711-...,ed87040c-...,f6e7827e-..." \
  --out /workspace/eval/videos/refcv4b_five_panel_step40284.mp4'
```

⭐ `--expect-step 40284` **refuses** a checkpoint whose own `step` differs — it does not mislabel.

**Content assertions, never the exit code** (a black video reports success identically):

```bash
python3 taniteval/tools/verify_mp4.py <out>.mp4     # ffprobe AND a real full decode + -count_frames
# plus: n PNG frames == expected, and each frame's mean pixel > 1.0
```

⚠️ `*.mp4` is git-ignored. The renderer, this runbook and representative stills are committed; the
mp4 is delivered as a file, by absolute path. There is a **30 MiB delivery gate** — ship the pair
(full + a `_web`/`_small` copy **re-encoded from the PNGs**, never transcoded).

### Two renderer defects found and fixed by the preflight (committed)

* the banner said **`refcv3 (REF-C v3, hier)`** and the sub-banner **"among its 128 anchors"** on a
  **refcv4b** frame whose own BEV panel read **117**. Identity is now DERIVED — `--run-label`
  (default: the checkpoint's own run directory) and `prov["n_anchors"]`. Verified in the rendered
  **pixels**: banner `refcv4b-b1-v72-40k (REF-C, hier)`, "its 117 anchors", legend "the 117-anchor fan".
* the nav caption hard-coded **`+0.0000 true−shuffled, MEASURED @ step 30 000`** — a **refcv3**
  number, printed where it reads as measured on the model being drawn. It now arrives via
  `--nav-shuffle-note`; the default says the quantity is not measured rather than borrowing one.

---

## 5. Compute etiquette

⛔ **Nothing above touches the GPU until GATE 0 passes**, because that is the moment the A40 stops
being a training pod. Cost probe + full eval + reel ≈ **1 h**. ⚠️ The PI has queued **refcv5**
training for this GPU — this is an eval-length job, not a long one, and the Master Mind is told
before anything longer is started on it.
