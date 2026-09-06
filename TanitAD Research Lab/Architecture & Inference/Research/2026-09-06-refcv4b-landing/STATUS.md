# refcv4b landing — STATUS (banked incrementally, 2026-09-05/06)

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv4b-landing/`
**Owner:** Arch+Inference FlyWheel · **Run:** `refcv4b-b1-v72-40k` on `tanitad-refcv3` (= `tanitad-a40`, A40)
**Run dir (pod):** `/workspace/experiments/refcv4b-b1-v72-40k/`

⛔ Nothing in this file is a driving result. The only model-inclusive numbers below are the
trainer's own in-training monitor at **T0 on 160 windows**, which `refc_v3_train.py:1553-1560`
itself stamps as *"NOT the four-metric-family result and must never be quoted as one"*.

---

## 1. P0 — the landing watch (ARMED, MEASURED)

| fact | value | source |
|---|---|---|
| target step | **40,284** (not 40,000) | `pod:/workspace/sup_refcv4b_v3.sh` `TARGET=40284` |
| step at 2026-09-05T22:27:33Z | 31,450 | probe marker |
| pace | **3.98 s/step** (124,625.2 s / 31,300 steps, in-log `elapsed_s`) | `metrics.jsonl` |
| ETA step 40,284 | **≈2026-09-06T08:00Z** | 8,834 steps remaining × 3.98 s |
| done-marker | written BY THE SUPERVISOR itself | `sup_refcv4b_v3.sh`: `echo '{"done": true, ...}' > $OUT/summary.json` on `last_step >= TARGET`, then `exit 0` |
| final checkpoint | **the rolling `ckpt.pt`** — `MILESTONES = (5000, 15000, 20000, 30000)` does NOT contain 40,284, but `refc_v3_train.py:1536` saves on `step % save_every == 0 **or step == args.steps**`, so 40,284 IS written to `ckpt.pt` (with optimizer, ~1.28 GB) | source read |

**The watcher** (`scratchpad/watch_refcv4b.sh` + `pod:/workspace/refcv4b_probe.sh`, md5
`69bd934bde45a603c32299865b61d71d`) exits on **completion AND on failure AND on channel death**:
`summary.json` present · error count ≥ 1 · step ≥ target without a marker · supervisor gone (×2) ·
trainer gone (×3) · 8 consecutive probe failures. It parses ONLY the opaque marker
`ZZ<step>-<sum>-<err>-<sup>-<trn>-<ckpt>ZZ`, which is **disjoint from every token it searches** —
the probe's own argv carries `T[r]aceback`, `CUDA out of m[e]mory`, `K[i]lled` … as character
classes, so no grep can match its own command line.

⭐ **The failure detector was proved to read non-zero.** A same-breath control file containing a
real `Traceback` + `CUDA out of memory` read **CONTROL_HITS=2**; `train.log` read
**690** matches for a benign pattern while the error count read **0**. So `errs=0` is a genuine
absence, not an unreadable file.

---

## 2. The training curve — MEASURED (T0 in-training monitor, n = 160 windows/eval, 63 evals)

**Artifact:** `raw/refcv4b_metrics_31700.jsonl` (pulled 2026-09-05T22:5xZ, md5
`f6a5ba913ba96ee40792fa58456be8e8`, byte-identical to the pod file), analysed by
`raw/evalcurve.py` and `raw/curve.py`.

⚠️ **n = 160 windows per eval** (8 batches × 20), a `manual_seed(12345)` loss monitor with no
`eid`. Every number in this section is therefore a **trend across 63 evals**, never a
decision-grade level. A 160-sample accuracy has 1σ ≈ 0.040.

### 2.1 ⛔ THE LONGITUDINAL kin3 HEAD IS WORSE THAN ITS OWN CLASS PRIOR

The core's 3-class maneuver heads are `loss_lat = F.cross_entropy(out["lat_logits"], lat_k)` /
`loss_lon` (`refc_v3_train.py:585-586`), labels `tac.window_factored_labels(pose_last,
fut_ext[:, :20])` — kin3, derived from poses. The **no-information floor** is the entropy of the
class prior, and the model carries that prior in its own checkpoint as
`core.lat_log_prior` / `core.lon_log_prior` (an EMA maintained by
`core.update_tactical_prior`, `refc.py:2061`).

Read from `ckpt_30000.pt` (CPU, `mmap=True`; CONTROL: 497 `core.` keys present, 551 tensors):

| head | class prior | **prior-predictor CE (nats)** | model `eval_*` @31,500 | verdict |
|---|---|---|---|---|
| lateral (kin3) | `[0.8361, 0.0754, 0.0885]` | **0.5591** | **0.4558** | beats the prior by **0.1033 nats** — i.e. it has taken **18.5 %** of the distance from prior to zero |
| longitudinal (kin3) | `[0.1777, 0.6451, 0.1771]` | **0.8964** | **1.0090** | ⛔ **WORSE than the prior by 0.1126 nats**, and only 0.0896 below *uniform* (ln 3 = 1.0986) |

⚠️ **Scope, stated honestly:** the prior is an EMA over the **TRAIN** marginals while the CE is on
**EVAL** windows, so the floor is approximate until the eval-set marginal is measured. That
measurement is a **required check in the landing eval** (§4), not a caveat to be carried.

⚠️ **This is NOT an inert diagnostic.** With `tac_vocab_version = "v7.0"` (this run's config),
`refc_v3.py:929` sets `man5 = None`, so H19's anchor prior falls back to **the core's own kin3
heads** (`refc.py:2294`, `:2313` — `lat_prior = log_softmax(lat_logits) − lat_log_prior`), which
`lat_to_anchor` / `lon_to_anchor` add to the anchor confidences (`refc.py:1578-1580`). ⇒ **a
longitudinal signal measured worse than a constant is being added to the selection surface.**

### 2.2 The lateral head converged by step ~1,000 and has not moved since

`eval_lat`: **0.4822 @1,000 → 0.4558 @31,500 = −5.5 % over 30,500 steps**, while over the same
span `eval_traj` went 1.2059 → 0.5902 (**−51 %**) and `eval_goal_tac` 17.3774 → 3.9250 (**−77 %**).

`eval_lon`'s **minimum over the entire run is at step 500** (0.8675) and it has never been beaten.

### 2.3 The strategic goal gate opens, slowly, and is still opening

`eval_goal_gate` rises **monotonically** 0.0008 @500 → **0.0658 @31,500**, gradient still non-zero
(`goal_gate_grad` last-20 median 0.0186). Read at ckpt_30000 the parameter is **0.06424**.
⇒ the strategic goal path contributes at ≈**6.6 %** amplitude after 31.5 k steps. Two readings —
found its optimum, or LR-bottlenecked — and they are **discriminable** (§5, refcv5 lever 3).

### 2.4 anchor_acc peaked mid-run

`eval_anchor_acc` **0.5312 @10,000 → 0.4562 @31,500**. ⚠️ At n = 160 that is ≈1.9σ and is **NOT
separated**; it is reported with its n precisely so it is not over-read. It is a hypothesis for the
landing eval, not a finding.

### 2.5 The run is at/near its plateau

`eval_loss` min **6.3510 @27,000**, last 6.4411 @31,500 — no improvement in 4,500 steps.
Last-5 vs prior-5 medians: 6 of 16 tracked eval metrics **worse or flat**.
⇒ ESTIMATED: the marginal value of steps 31.5 k → 40.3 k is small. This is a *prediction the
landing eval tests*, since `ckpt_30000.pt` is banked and comparable.

---

## 3. P1 readiness — the eval path is GREEN, and it was NOT green when I started

### 3.1 ⛔ The pod was missing 137 of the 224 taniteval files, including the STRATEGIC module

MEASURED by full md5 manifest, both sides:

| | count |
|---|---|
| repo `taniteval/**/*.py` | 224 |
| pod, before | 87 (a strict SUBSET — **0 pod-only files**, so nothing could be lost by syncing) |
| missing on pod | **137**, incl. `taniteval/taniteval/nav_compliance.py`, `tools/verify_mp4.py`, `tools/render_refav1_video.py` |
| pod `refcv3_arm.py` | 2,127 lines vs repo **2,907** — the pod copy predates commits `ae9a762` / `2198e3e` (the strategic-family fix) and `5cd86fd` (`ha0_ext`) |

Synced by `scp` + tar; verified by md5 manifest: **210/210 shipped files identical, 0
mismatches**, the 14 not shipped are all under `taniteval/results/` (deliberately excluded). Pod
backup at `pod:/workspace/taniteval.BAK-20260906`.

⚠️ **CORRECTION to a premise I was given and passed on.** *"The STRATEGIC family was silently
absent from every refcv3 eval for its entire life"* is **too strong**, and a re-analysis of the
banked dump refuted it: `refcv3-40284-openloop.ARM.json` **already carries** the route-head
strategic block (n = 3,622, accuracy 0.7667, κ 0.4604 [0.3775, 0.5432]). What is absent from the
banked records is the **`nav_compliance` sub-block** — and the labels-dict fix does **not**
recover it, because `nav_compliance.py:1046-1056` returns on missing sidecar keys **before**
`resolve_labels_path` is ever reached, and six required keys (`plan_full_nav_true`,
`gstr_nav_true`, `gt_future_ext`, `pose_last`, `ego_t0`, `ep_poses`) are absent from **every**
episode of **both** banked dumps.
⇒ **A re-analysis cannot recover it; a fresh roll can.** My preflight — a fresh roll with the
synced tool — produced a fully populated `nav_compliance`. So the landing eval will deliver a
strategic sub-block **that refcv3's banked record does not have**, and a like-for-like refcv3
comparison on that sub-block needs a refcv3 GPU re-roll (a named, costed work item, not a caveat).

### 3.2 ⛔ The model code was NOT synced, on purpose

The repo's `stack/tanitad/refs/refc.py` is **+589 / −20 lines** against the pod's (after CRLF
normalisation — the raw diff aligns nothing because the POD file is CRLF and the repo file is LF).
Those additions are a **sibling agent's uncommitted refcv5 WP-4 sampler / WP-6 agent-seam wiring**
(`control_head`, `agent_gate`, `norm_a`, `cross_agent`). `refc_v3.py` differs by a purely additive
`withheld_speed` / `bank_speed_pred` plumbing (all new args default `None`).

⇒ **The eval rebuilds the model with the POD's `refc.py`/`refc_v3.py` — the code that actually
trained.** Evaluating refcv4b under a tree carrying unmerged refcv5 wiring would be the
pod-drift trap with the arrow reversed.

### 3.3 The full 7-arm path ran end to end, CPU, on the pod

`ckpt_30000.pt`, 1 clip, stride 40, **5 windows**, `--device cpu`, exit 0. Arms produced:
`os · ha · ha0 · ha0_ext · os_navshuf · os_navzero · oracle_sel`.

⭐ **`ha0_ext` IS wired** — `refcv3_arm.py:1399` reads `arms = ["os","ha","ha0","ha0_ext"]`. The
claim that it is unwired (`REFCV5_DESIGN_PLAN.md` §E2, register `D-RUNGA1`) is **STALE against the
current tree**; the code read supersedes the doc read.

⭐ **STRATEGIC is populated**: `refcv3.strategic` carries `n_windows`, `n_route_labeled`,
`nav_shuffle`, three `conditionings` with per-class recall/precision + confusion + kappa, **and a
populated `nav_compliance` sub-block** (readouts `plan` and `gstr`, tolerances, per-conditioning
compliance with episode-cluster bootstrap). `_defects` is `null` and a recursive scan finds no
`defect: true` anywhere. CONTROL: `grep -c ha0_ext` on the same file reads **24**, so the single
`nav_compliance` hit is a real read, not an unreadable file.

⚠️ `families_unavailable=['strategic']` in the per-arm line is **BY DESIGN and is not the bug** —
`four_families.strategic` needs `route_pred`/`route_gt` off a trajectory, which a short path cannot
carry (`refcv3_arm.py:2413-2416`). refcv3/v4's real strategic family is the route-head +
nav-compliance block above.

⚠️ `ol` is **ABSENT WITH ITS REASON**, not dropped: refcv3/v4 consume no recorded actions, so
integrating them is a corpus property, not a rollout of this model (`refcv3_arm.py:26`).
⛔ `oracle_s0` is **not an arm anywhere in the code** — it is a refav1 *run directory name*. The
refcv3/v4 T0 ceiling is `oracle_sel`.

### 3.4 Shipped and md5-verified on the pod

| artifact | pod path | md5 |
|---|---|---|
| lead block (LONGITUDINAL distance-keeping) | `/workspace/eval/b1_eval_lead_block.npz` | `33a48e15a52eb9dbd69ecd6026fd5023` |
| 9-clip extrinsics union | `/workspace/eval/extrinsics_union9.json` | `dd8b3d21cb717a0e18714211d4d914db` |
| landing probe | `/workspace/refcv4b_probe.sh` | `69bd934bde45a603c32299865b61d71d` |

The lead block joined cleanly (`speed-check max 7e-06 m/s`, `NO_LABEL 0`); `distance_keeping`
read REFUSED only because all 5 preflight windows were `NOT_STRAIGHT` — an artifact of a 1-clip
probe, not a defect (refcv3's full read had n = 1,252 windows / 67 episodes).

⭐ **`scp` works on the pod's DIRECT port** (1.48 MB in 3.5 s incl. handshake). The standing note
that the RunPod path cannot move files applies to the `ssh.runpod.io` **proxy**, not to
`$HOST:$PORT`.

---

## 4. The landing eval — required checks (each is a FAILURE if absent, not an omission)

1. GATE 0: `summary.json` `{"done": true}` · supervisor process count reads **0** ·
   `ckpt.pt` copied to an immutable name **and md5'd** before anything reads it.
2. All four families, never pooled, each with the **paired episode-cluster bootstrap**
   (`taniteval/ci.py::paired_episode_cluster_bootstrap`), its CI and its **T-tier** stamp.
   ⛔ `overlapping_holdout_se` is forbidden.
3. **STRATEGIC present and non-empty**, asserted positively with a control that must read a
   known value.
4. Controls: `ha0` (constant velocity) · `ha` (hold-action) · **`ha0_ext` (the echo control)** ·
   `os_navshuf` and `os_navzero` (both, BACKLOG R39) · `oracle_sel` at T0.
5. **Turn recall per class beside ADE** — refcv3 @40,284 reads `turn_left` 0.8048 /
   `turn_right` 0.8636 / `lane_keep` 0.9715 (κ 0.8113); longitudinal `brake_stop` 0.3835 /
   `accelerate` 0.3439 (κ 0.3078). refav1's best-ADE arm executes **zero** turns, which ADE cannot see.
6. **The eval-set kin3 class marginal**, to convert §2.1's floor from train-EMA to measured.
7. `TanitAD_BenchmarkCriteria` over the artifact; every missing criterion reported as a WORK ITEM.
8. The trivial profile and the selection profile FIRST — ⛔ if either is degenerate the read is
   **VOID, not negative**.

**The pre-registered first gate** (`MODEL_REGISTRY.md` §4.6, `D-REFCV4-GATE1`):
`oracle_sel ≥ 0.2996 separated ⇒ kill`. ⚠️ In the 5-window preflight `oracle_sel` read **0.7941**,
*worse* than `os` 0.348 — the opposite order to refcv3 (`oracle_sel` 0.3668 < `os` 0.4419).
At n = 5 this is not evidence; it is the **first thing the landing eval must resolve**.

---

## 5. Incumbent baseline refcv4b must beat (refcv3 @ 40,284, T1/UNRULED, n = 4,823 win / 141 eps)

| arm | ADE (m) | FDE (m) |
|---|---|---|
| `ha` hold-action | **0.2996** [0.2755, 0.3278] | 0.6588 |
| `oracle_sel` (T0 ceiling) | 0.3668 | 0.7770 |
| **`os`** the model | **0.4419** [0.4098, 0.4743] | 0.9288 |
| `os_navzero` (deployment) | 0.4659 | 0.9655 |
| `ha0` const-velocity floor | 0.6723 | 1.4029 |

⛔ `os − ha` = **+0.1423 [+0.1187, +0.1658] — refcv3 LOSES to a trivial hold-action control**,
and the loss is **2.87× larger on manoeuvre windows**.
Source: `MODEL_REGISTRY.md` §4.5 lines 2742-2759, raw
`taniteval/results/refcv3-40284-openloop.json`.

---

## 6. The FULL eval chain is preflighted, and it already closes 3 of refcv3's criteria violations

The landing chain is **two** commands, not one, and both have now been run end to end on
`ckpt_30000.pt`:

```
refcv3_arm.py  --dump-dir D --out A.json          # the arm record + the dump
openloop_suite.py --arm-json A.json --dump-dir D  # the CRITERIA-CHECKABLE artifact (ZERO GPU)
```

⛔ **The second command is not optional, and finding that out was the point of preflighting it.**
`tools/criteria_check.py` on the `refcv3_arm.py` record alone returns

> `UNKNOWN_SCOPE: matches no in-scope or out-of-scope marker — the driving criteria do not apply
> to this artifact — not scored.`

⇒ a refcv4b record in that shape would have **silently escaped the binding completeness check**,
and `UNKNOWN` *is never counted as compliant*. This is the same failure the criteria skill already
retracts for the refav1 shape ("matched no in-scope marker, and every one read `UNKNOWN_SCOPE`").
The `openloop_suite` shape **is** in scope, and it is how `refcv3-40284-openloop.json` was made.

**MEASURED, both under registry v2.8.0:**

| artifact | violations | work items |
|---|---|---|
| `taniteval/results/refcv3-40284-openloop.json` (the incumbent baseline) | **3** | 0 |
| refcv4b preflight, same chain | **0** | 1 |

refcv3's three violations are **exactly the nav-compliance block** — the criterion and its two
controls (paired drop under nav-SHUFFLE, paired drop under nav-ZERO). They are absent there because
that record predates the module; my taniteval sync ships it, and the preflight produces it
populated. ⇒ **the refcv4b record will be the first REF-C record with zero criteria violations**,
and that is attributable to the sync, not to the model.

⚠️ **Two invocation traps the preflight caught, both silent:**
1. **`--dump-dir` must be passed ALONGSIDE `--arm-json`.** Without it the suite refuses the
   `const0` constant-only control and prints *"⛔ THE HARNESS IS WRONG, NOT THE MODEL"* — which
   reads like a defect and is actually a missing flag. With it: **`constant-only control: OK`**,
   i.e. all three exact checks pass (paired-against-itself bit-exactly 0/0/0; `const0`'s ADE equals
   the mean GT displacement recomputed independently in float64 numpy; its speed MAE equals the
   mean GT speed). That control is the only thing that catches a metric whose normalisation is
   wrong, so a suite without it has no check that the pipeline reads a KNOWN value.
2. **The pod's criteria registry was v2.5.0 against the repo's v2.8.0.** Shipped and md5-verified
   (`c4b5f4932ab247def1addc1f13d76371`), along with `tools/criteria_check.py`.

**The one remaining WORK ITEM — reported, not papered over:** `hyg.inference_seed`
(*"stochastic planner: inference-seed replicate + the seed floor"*, `required: false`,
`applies_when: the scored arm's planner SAMPLES at inference`). refcv4b's decoder is
**deterministic at eval** — `refc.py:1596` uses `zeros_like` for the noise when not training, and
the DiffusionDrive audit measured `eval_deterministic_steps0/2: true` — so the criterion does not
apply and the admissible state is **REFUSED with that reason**, not ABSENT.
⇒ **Work item for the Eval FlyWheel** (whose instrument this is): emit
`refused["inference_seed_replicate"]` with the determinism reason when the decoder's eval noise is
`zeros_like`. I am not editing another FlyWheel's working instrument to move a `required: false`
counter; this is escalated in the report instead of being written into a README nobody re-reads.

## 7. The reel is bigger than planned, because its stated constraint was false

See `RUNBOOK.md` §4. Short form: the calibration bound was not real, the reel is selected by a
**rule** over all 141 held-out clips rather than by which 9 happened to have extrinsics, and it
lands at **15 clips / 2,571 frames / 257.1 s — 4.95× the banked refcv3 reel**, with 5 net-left and
6 net-right clips, 7 reaching a full stop and 6 exceeding 15 m/s.
