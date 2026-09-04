# refcv3 @ step 40,284 — the epoch-end open-loop read

**Arm:** `refcv3-b1-v72-30k` (`--arm hier --size base`, 40,284 steps) · **pod:** `tanitad-refcv3`
(A40, 69.30.85.211:22001) · **instrument:** `taniteval/tools/openloop_suite.py` over
`taniteval/tools/refcv3_arm.py` → `t1_eval.analyze` · **written:** 2026-09-03/04 ·
**owner:** Architecture & Inference FlyWheel (final-checkpoint eval stream)

> ⛔ **EVERYTHING HERE IS OPEN LOOP** (PI ruling 2026-09-02). No arm is scored in a simulator and
> none of them steers anything. The model does not control the vehicle; the ego data keeps arriving
> from the eval recording.

---

## §0 STATUS — ⭐ **COMPLETE. The read ran 2026-09-04 03:56–04:10 UTC on the freed pod.**

⛔ **LINE ONE — THE TRIVIAL CONTROL STILL WINS.** At step 40,284 the hold-action control `ha`
beats the deployed arm `os` on ADE by **+0.1423 m [+0.1187, +0.1658]**, paired episode-cluster
bootstrap, separated **the wrong way**. The extra 10,284 steps past the 30,000 milestone closed
**21 %** of that gap (30k: `+0.1803 [+0.1563, +0.2047]`) and did **not** close it. The 30k
headline stands at the epoch: `ha` **0.2996** < `os` **0.4419** < `ha0` **0.6723**.

⭐ What DID improve is the margin over the constant-velocity floor: `os − ha0` went
**−0.1924 → −0.2304** and `os` itself **0.4799 → 0.4419 m**. `ha` and `ha0` are checkpoint-
independent and read **bit-identically** at both steps on the same 4,823 windows, which is the
internal control that the two reads are on the same surface.

**Checkpoint step VERIFIED BY CONTENT: `ckpt["step"] == 40284`** (`--expect-step 40284` also
enforced it inside the suite). `MEASURED (ours)`; §3 and §6 below.

---

## §1 GATE 3 — train ∩ eval, **MEASURED BY EPISODE ID** (not arithmetic)

`config.json`'s `v2_parity.checked` is **false**, so the trainer verified nothing. The
`4,572 = 4,713 − 141` identity that has been quoted is an *arithmetic* consistency check, not a
disjointness check. This is the disjointness check, and it **PASSES**.

| quantity | value |
|---|---|
| `.v2ep.pt` clip ids in `/root/data/train` | **4,572** |
| `.v2ep.pt` clip ids in `/root/data/eval` | **141** |
| **intersection** | ⭐ **0** |
| union | **4,713** |

`MEASURED (ours)` 2026-09-03, `scratchpad/gate3_disjoint.py` on the pod; marker `ZZG3-4572-141-0-4713ZZ`.

**Labels.** `data/s2_labels_v7.2_eval.jsonl.gz` md5 **`aa12c948f062181c3297265b51526ec5`** — matches
the canonical value stamped in `config.json::nav_from_v7_stats`. 147 records, `split` all `eval`,
`release` all `v7.2`; **141 of the 147 carry pixels** in the B1 epcache and are the scored set — the
other 6 are the deployed-val40 clips the parity gate drops.

**⭐ The nav token is an ORACLE, confirmed at the label file itself.** Every `nav_command` record
carries `provenance: 'ego-future'` and (where it commands a turn) `oracle: True`. The route input
is optimistic by construction and **will not exist at deployment** ⇒ `os_navzero` gets equal
billing with `os`, not a footnote.

## §2 NON-PARITY — MEASURED from the run's own `config.json`

```
v2_parity = {parity: false, checked: false, corpus_key: null,
             cache_dirs: ["/root/data/train"], uid_kind: "v2ep_clipid",
             label: "v3 v2-cache", clips_present: 4572}
```

⇒ **refcv3 is NOT cross-arm comparable at the LEVEL** with `refc-base` / `refc-xl`. What *does*
travel across a parity break is **each arm's margin over its own `ha0`**, because `ha0` is
bit-identically defined across models (`a = 0`, `κ = 0` at the measured `v0`, same integrator, same
grid). A level against another model's level does not.

## §3 GATE 0 / GATE 1 — the three conditions, MEASURED 2026-09-04

| gate | check | reading |
|---|---|---|
| GATE 0 | `summary.json` | `{"done": true, "final_step": 40284, "target": 40284}` |
| GATE 0 | save-before-eval order in `train.log` | `[v3:hier] ckpt step 40284 -> ckpt.pt` **precedes** `[v3:eval] step 40284`, then `DONE at 40284 — summary.json written` |
| GATE 0 | trainer alive | `ps -eo args \| grep -c '[r]efc_v3_train'` = **0** |
| GATE 0 | supervisor alive | `ps -eo args \| grep -c 'sup[_]refcv3'` = **0** (only a jupyter-lab remains on the box) |
| GATE 0 | immutable copy | `cp ckpt.pt ckpt_40284_FINAL.pt` — both md5 **`fc304b62686ddb9e685d14bdab482404`**, 1,284,991,701 B |
| GATE 0 | ⭐ **step BY CONTENT** | `torch.load(...)["step"]` → **40284**; keys `['model', 'opt', 'step']`. ⚠️ `ckpt_40284_FINAL.pt` is a copy *I* made — `MILESTONES = (5000, 15000, 20000, 30000)` never writes it, so its NAME proves nothing and `--expect-step 40284` is what proves it |
| GATE 1 | pod free | A40 `0 MiB / 46,068 MiB` used, `0 %` util before launch |
| GATE 1 | disk | real `dd`: 512 MiB at **355 MB/s** (⛔ never `df` — it reports the 965 TB cluster) |
| GATE 1 | `OMP_NUM_THREADS=6` | set on every invocation |
| GATE 1 | banked dump reused | the `--analyze-only` re-read of the dump cost **zero GPU** |

### ⛔ §3.1 A SECOND STALE-STACK DEFECT THE PROBE CAUGHT — one module deeper than §4.1

§4.1 records the preflight catching `stack/tanitad/models/kinematic.py` missing
`STEER_WHEELBASE_M`. **That was not the only one.** The 2-episode probe (§"Step 1" of
`OPENLOOP_SUITE.md`) died on the MAIN path at:

```
File "taniteval/tools/refav1_arm.py", line 440, in paths_from_controls
    return unicycle_paths(c[:, :k].float(), v0_t, dt, action_units=action_units)
TypeError: unicycle_paths() got an unexpected keyword argument 'action_units'
```

The pod's `stack/tanitad/refs/refa_v1_plan.py` predated commit `56fdea0` (*"The steer/curvature
interface is resolved"*): its `unicycle_paths(controls, v0, dt)` had **no** `action_units`/
`wheelbase` parameters, so `--action-units steer` — the repaired kinematic contract — could not be
applied at all. The consumer (`kinematic.py`) had been updated; the **caller** had not.

Repaired by shipping the repo file (LF) and verifying on the far side by **md5**
(`5f18acf56126d18fb2d4619dbc750434`, 17,263 B) **and** by grepping the fix out of the pod-side
copy (`:293 action_units: str = "kappa"`, `:316 controls = as_curvature(...)`), then purging the
stale `__pycache__`. *Root-cause class: the same silent pod-stack drift as §4.1 — and it shows
that fixing the module a symbol lives in does not fix the module that calls it.*

⭐ **A full normalised-md5 sweep of `stack/tanitad/` (150 repo files vs 132 pod files) was then run
so this could not happen a third time mid-run.** Line endings had to be normalised first: the repo
checkout is CRLF and the pod is LF, which made **8** files look drifted when only **7** were —
`models/kinematic.py` is byte-identical once normalised. The 7 truly drifted:
`data/parity.py`, `data/refav1_loader.py`, `models/metric_dynamics.py`, `refs/refa_v1.py`,
`refs/refa_v1_plan.py` (**shipped**), `refs/refc.py`, `refs/refc_v3.py`. ⛔ **The last two were
deliberately NOT shipped** — the pod's copies are the tree that TRAINED this checkpoint and they
load it correctly; replacing a model module before reading its own checkpoint would change the
architecture under the weights. The other four are not on this eval's import path (proved by the
probe running the full path to a rendered report).

⚠️ 18 repo files are ABSENT on the pod (`eval/echo_gate.py` + the whole `lake/` package). None is
imported by this eval; recorded so the absence is visible rather than discovered later.

## §4 PREFLIGHT — what had to be shipped, and the one real defect it caught

⛔ **The pod had NO `taniteval/` at all** — `/workspace/TanitAD/` held only `data/` and `stack/`,
and is not a git repo. Neither `openloop_suite.py` nor `refcv3_arm.py` was present. **Nothing could
have been launched from that pod as it stood.**

⛔ **Pods have no git credentials** — `git fetch` there HANGS and a checkout would RESET the tree to
an ancient commit, destroying shipped files. Everything below arrived by **file-ship with per-file
md5 verification at both ends**, never by git.

| shipped | how verified |
|---|---|
| 94 files: the `taniteval` package + `taniteval/tools/*` + `tools/criteria_check.py` + `products/P7-TanitEval/CRITERIA_REGISTRY.json` | tar md5 `4d9b19a5080d656626e40f098cad9029` both ends; then **per-file md5 recomputed pod-side against the shipped manifest → `ZZMD5-94-0-0ZZ` (94 OK, 0 mismatched, 0 missing)** |
| `b1_eval_lead_block.npz` → `/workspace/eval/` | md5 `33a48e15a52eb9dbd69ecd6026fd5023` both ends |

**Both of the fixes made to `refcv3_arm.py` today were grep-verified in the POD-SIDE file**, not
assumed from the fact that a file exists:

* **(a) the CUDA/CPU argmin** — `refcv3_arm.py:968` reads
  `anchors_bank = model.core.decoder.anchors.detach().cpu()`, and both consumers (`:1054`, `:1062`)
  cast the target to `anchors_bank.dtype`. This is on the MAIN path, not only under
  `--with-oracle-sel`.
* **(b) the 10 Hz lead-block key** — `:1353` records that `raw // 2` plus the default `t → 2t` map
  *"truncated every ODD frame to frame−1"*, and `:1360` states `join_lead_block` now takes a row per
  RAW frame. That defect silently deleted the ENTIRE distance-keeping family.

**Disk:** real `dd` write test (⛔ never `df`, which reports the cluster and hides the per-pod
MooseFS quota) — **600 MiB at 544 MB/s**, then removed.

### ⭐ §4.1 THE DEFECT THE PREFLIGHT IMPORT PROBE CAUGHT

A CPU-only, GPU-free probe of `rebuild_config` against the live run's own `config.json` died at
**import**:

```
ImportError: cannot import name 'STEER_WHEELBASE_M' from 'tanitad.models.kinematic'
             (/workspace/TanitAD/stack/tanitad/models/kinematic.py)
```

The pod's `stack/tanitad/models/kinematic.py` was **556 lines against the repo's 637** and contained
**zero** occurrences of `WHEELBASE`. `refav1_arm.py:167` imports that symbol at module level, and
`refcv3_arm.py` imports `refav1_arm` — so **every invocation of the suite would have died at import**.

Diffed before shipping: the repo file is a **strict superset** — `0` pod-only lines, `+81` repo-only
lines, all of them the command↔geometry bridge (`STEER_WHEELBASE_M = 2.9`, `kappa_of_steer`,
`steer_of_kappa`, `as_curvature`). refcv3's model modules (`refc_v3.py`, `refc.py`) do **not** import
`kinematic`, so the update cannot change the rebuilt architecture. Shipped **atomically** (write to
`/tmp`, then `mv` — never a truncated file under a live trainer), backup left at
`kinematic.py.bak-preRead`. Verified after: `ZZSYM-2.9ZZ`.

*Root-cause class: **a pod stack that drifts silently, where a launch from it resurrects a fixed bug**
— and the countermeasure that worked is the one `CLAUDE.md` prescribes: a **preflight import probe**,
which turned a failure-after-the-expensive-part into a two-second one.*

### §4.2 Currency of the pod's stack — what was NOT changed, and why

⚠️ **Presence proves transfer, md5 proves bytes, a successful import proves loading — none of them
proves currency.** The runtime import closure was md5'd pod-side and content-diffed against the repo:

| module | verdict |
|---|---|
| `geometry.py`, `refs/refb.py`, `refs/refc_tactical.py`, `models/vocab_v7.py` | **identical in content** (0 diff lines); the md5 differs only because the repo checkout is CRLF and the pod is LF |
| `models/metric_dynamics.py` | differs **only** by the `bptt_truncate` training feature (default `0` = byte-identical); its own docstring states the forward pass is unchanged and every returned pair value-identical ⇒ **inert at eval** |
| `data/refav1_loader.py` | pod lacks `out["v0"]`. ⭐ **Inert for this read**: `refcv3_arm.py:995` derives `v0 = float(pose_last[3])` itself and never takes it from the loader. **Left untouched** rather than perturbing a module the live trainer imports. |
| `config.py`, `data/v2_dataset.py`, `data/physicalai.py`, `refs/refc.py`, `refs/refc_v3.py`, `models/v6.py`, `eval/v6_probe_trunk.py` | stale vs repo and **DELIBERATELY NOT UPDATED** — this is the tree that **trained** the model, and the model must be rebuilt from the tree that trained it |

**`rebuild_config` then MEASURED working on the pod against the live `config.json`**, returning
`arm hier`, `steps 40284`, `image_hw [256, 640]`, `tac_vocab_version v7.0`, `hier True`, with
`tanitad` and `refc_v3` both resolving from `/workspace/TanitAD/stack`. The full module-level import
closure of the suite was proved by a successful `--help` on the pod; `criteria_check.py` +
`CRITERIA_REGISTRY.json` **v2.5.0** load there too.

## §5 STRATEGIC — why the family is structurally unavailable on this corpus

⛔ **Do not read a route-head probe as the STRATEGIC family.** At step 30,000 the artifact carried
`status: UNAVAILABLE, n: 0` on **every** arm, and a *separate* `/refcv3/strategic/` route-head probe
was quoted in prose as though it were the family. It is not, and this read will not repeat that.

Whether the family can be populated is now settled at source. `how_to_populate` asks for
map-derived option sets from `stack/experiments/nurec-gsplat/strategic_gt.py`, which emits them
**"from `map.xodr` + the clipgt ego track"** — an **OpenDRIVE HD map**, a NuRec/AlpaSim asset.
PhysicalAI-AV ships **no map, lane graph, junction annotation or route signal** (the dataset card
says so verbatim), and its `egomotion` carries no lat/lon, so OSM map-matching on our traces is
impossible. Its own module states the reason the substitute is refused: *a route label read off the
ego's own future yaw is circular — it cannot distinguish "took the left branch" from "drifted left
on a curving road"*, and only poses with ≥2 map-admitted options are scoreable at all.

⇒ **STRATEGIC is UNAVAILABLE for the PhysicalAI B1 eval slice for a structural reason, not for want
of effort.** It is reported as UNAVAILABLE with its `n`, its `reason` and its `how_to_populate`
quoted from the JSON. The absence stays visible.

## §6 THE FOUR FAMILIES — MEASURED, step 40,284

**Instrument:** `taniteval/tools/openloop_suite.py` over `taniteval/tools/refcv3_arm.py` →
`t1_eval.analyze`. **Run:** 2026-09-04T03:56–04:10Z, pod `tanitad-refcv3` (A40).
**n = 4,823 windows / 141 episodes**, `--window-stride 5`, grid `2s`
(K=4, instants 0.5/1.0/1.5/2.0 s ← model slots 5/10/15/20), `--action-units steer`.
**Estimator:** paired episode-cluster bootstrap (`taniteval/ci.py`), `n_boot 2000`, `seed 0`,
cluster unit = the episode. ⛔ `overlapping_holdout_se` is not used anywhere.
**Tier `T1`, ruling OPEN.** ⛔ **Regime: OPEN LOOP on every arm** (PI, 2026-09-02).
**Criteria checker:** registry v2.5.0 — **0 violations, 0 work items.**
**Harness controls:** `const0` self-paired `0.0 [0.0, 0.0]` **bit-exact**; `const0` ADE
`14.248286` vs `14.248286` expected (|Δ| 1.07e-07); `const0` LON speed MAE `11.395674` vs
`11.395674` (|Δ| 0). **PASS** — the harness may be read.

### 6.1 ⛔ THE HEADLINE: the arm levels, with BOTH controls beside them

| arm | what it is | ADE (m) | FDE (m) |
|---|---|---|---|
| `ha` | **hold-action control** (the (a, steer) closing at t0, held) | ⭐ **0.2996** [0.2755, 0.3278] | **0.6588** [0.6044, 0.7192] |
| `oracle_sel` | **T0 ceiling** — GT-nearest anchor (`a_star`). ⛔ not a driveable arm | 0.3668 [0.3437, 0.3914] | 0.7770 [0.7261, 0.8297] |
| **`os`** | **the deployed arm** — model's own `sel_score_v3` selection, ORACLE nav | **0.4419** [0.4098, 0.4743] | **0.9288** [0.8611, 0.9947] |
| `os_navshuf` | nav pairing broken, marginal preserved | 0.4563 [0.4240, 0.4884] | 0.9582 [0.8887, 1.0246] |
| **`os_navzero`** | ⭐ **the DEPLOYMENT condition** — nav withheld (`nav_cmd=None`) | **0.4659** [0.4310, 0.5010] | 0.9655 [0.8935, 1.0362] |
| `ha0` | **constant-velocity control** (a = 0, κ = 0 at the measured v0) | 0.6723 [0.6007, 0.7469] | 1.4029 [1.2484, 1.5646] |

**Paired contrasts on the SAME windows** (`Δ`, 95 % CI, separated?):

| contrast | ADE Δ | verdict |
|---|---|---|
| `os − ha0` | **−0.2304 [−0.2881, −0.1781]** sep | **WON** — beats constant velocity |
| `os_navzero − ha0` | **−0.2064 [−0.2673, −0.1479]** sep | **WON** — and survives nav withdrawal |
| ⛔ **`os − ha`** | ⛔ **+0.1423 [+0.1187, +0.1658]** sep | ⛔ **LOST — the trivial hold-action control WINS** |
| `ha − ha0` | −0.3727 [−0.4346, −0.3161] sep | the control that must be beaten is itself far below the floor |
| `os − os_navshuf` | −0.0144 [−0.0220, −0.0069] sep | the model does use *this* window's nav — but the effect is 6 % of the `ha` gap |
| `os − os_navzero` | −0.0239 [−0.0428, −0.0089] sep | withholding nav costs 0.024 m; a **LOWER BOUND** (the core collapses to `follow`, it does not lose the input) |
| `oracle_sel − os` | −0.0751 [−0.0884, −0.0618] sep | the **selection gap**: perfect anchor choice would buy 0.075 m, i.e. it is not where the 0.14 m lives |

⭐ **Read the three together.** Oracle *selection* is worth 0.075 m and oracle *nav* 0.024 m —
together **0.099 m**, still short of the **0.142 m** by which the hold-action control wins. Even a
refcv3 with a perfect anchor chooser AND its oracle route would not reach `ha` on this surface.

### 6.2 vs step 30,000 — what the last 10,284 steps bought

Both reads are on the **same 4,823 windows / 141 episodes** with the same instrument and grid, so
these are step-matched. `ha` and `ha0` do not depend on the checkpoint and read **bit-identically**
at both steps — that identity is the control proving the two reads share a surface.

| quantity | step 30,000 | step 40,284 | change |
|---|---|---|---|
| `os` ADE | 0.4799 | **0.4419** | **−0.0380 m** better |
| `os − ha0` | −0.1924 [−0.2525, −0.1379] | **−0.2304 [−0.2881, −0.1781]** | margin over CV grew |
| ⛔ `os − ha` | +0.1803 [+0.1563, +0.2047] | ⛔ **+0.1423 [+0.1187, +0.1658]** | gap to hold-action closed by **21 %**, still separated the wrong way |
| `ha` ADE | 0.2996 | 0.2996 | identical (control) |
| `ha0` ADE | 0.6723 | 0.6723 | identical (control) |

⇒ **The epoch improved the model without changing the verdict.** Training longer is moving in the
right direction at a rate that would need roughly another ~3.7 epochs of the same slope merely to
reach parity with a control that costs nothing — and that extrapolation is `ESTIMATED` from **two
points**, so it is a scale statement, not a plan.

### 6.3 LONGITUDINAL

**Absolute (arm `os`, n = 4,823):** speed MAE **0.4516** [0.4197, 0.4828] m/s · speed bias 0.0327
[−0.0106, 0.0744] · speed RMSE 0.7384 [0.6836, 0.7917] · along-track MAE **0.4030** [0.3722, 0.4332] m
· along bias 0.0372 [−0.0047, 0.0799] · accel MAE **0.6806** [0.6464, 0.7156] m/s² ·
target-speed accuracy within 0.5/1.0/2.0 m/s = **0.7135 / 0.8784 / 0.9713** ·
ego progress ratio mean **1.0031** [0.9886, 1.0144], under-progress rate 0.4915.

**Distance-keeping (headway / time-gap / TTC to the lead agent)** — ⭐ **PRESENT, and this is the
family the `join_lead_block` 10 Hz fix restored**: coverage over 4,823 windows / 141 episodes =
LEAD **1,489** · NO_LEAD 1,145 · NOT_STRAIGHT 1,889 · NO_LABEL 300, speed-check max 2.6e-05 m/s.
`os`: mean min headway **28.0645 m** (n 1,252), mean min time-gap **3.9744 s** (n 1,160),
mean min TTC **24.9877 s** (n 464 closing; **788 of 1,252 windows never close and are censored at
TTC_CAP_S = 30 s — quote `n_closing` beside the mean**).

| metric | `os − ha0` | `os − ha` |
|---|---|---|
| min headway (m) | +0.0606 [−0.0493, +0.1847] **TIED** | ⛔ −0.1280 [−0.2295, −0.0293] **LOST** (closer than the control) |
| min time-gap (s) | +0.0278 [−0.0014, +0.0622] **TIED** | −0.0167 [−0.0350, +0.0013] **TIED** |
| min TTC (s) | **+0.6717 [+0.1706, +1.2360] WON** | — |

| paired metric | `os − ha0` (oracle nav) | `os_navzero − ha0` (deployment) | ⛔ `os − ha` |
|---|---|---|---|
| target-speed MAE (m/s) | −0.0364 [−0.0636, −0.0079] **WON** | −0.0134 [−0.0437, +0.0179] **TIED** | +0.1976 [+0.1737, +0.2204] **LOST** |
| along-track MAE (m) | −0.0674 [−0.0923, −0.0415] **WON** | −0.0429 [−0.0715, −0.0117] **WON** | +0.1682 [+0.1441, +0.1923] **LOST** |
| acceleration MAE (m/s²) | ⛔ +0.2020 [+0.1713, +0.2342] **LOST** | ⛔ +0.2226 [+0.1893, +0.2561] **LOST** | +0.3640 [+0.3381, +0.3889] **LOST** |

⇒ **family verdict vs `ha0`: LOST (2 WON / 1 LOST); deployment: LOST (1 WON / 1 LOST / 1 TIED);
vs `ha`: LOST 3/3.** The arm is positionally better than a straight line but **less smooth** than
one, and target-speed accuracy is the first thing that dies when the oracle nav is withheld.

### 6.4 LATERAL

**Absolute (arm `os`):** heading MAE **1.3109°** [0.8087, 2.2671] (n_steps 18,115) · yaw-rate MAE
**1.794 °/s** [1.5761, 2.0309] · curvature MAE **0.008815 1/m** [0.006313, 0.012054] · curvature
bias −0.0002 · cross-track MAE **0.1084 m** [0.0958, 0.1222] · cross-track final MAE 0.2337 m ·
excluded below `min_ds` 1,177.

| paired metric | `os − ha0` | `os_navzero − ha0` (deployment) | ⛔ `os − ha` |
|---|---|---|---|
| cross-track MAE (m) | −0.2048 [−0.2686, −0.1500] **WON** | −0.2034 [−0.2669, −0.1489] **WON** | −0.0142 [−0.0288, +0.0004] **TIED** |
| heading MAE (deg) | −1.3741 [−1.7569, −1.0418] **WON** | −1.3562 [−1.7376, −1.0191] **WON** | −0.1332 [−0.2539, −0.0138] **WON** |
| yaw-rate MAE (rad/s) | ⛔ +0.1700 [+0.1006, +0.2534] **LOST** | ⛔ +0.1957 [+0.1186, +0.2864] **LOST** | +0.1849 [+0.1162, +0.2683] **LOST** |

⇒ **family verdict vs `ha0`: LOST (2 WON / 1 LOST), same in deployment.** ⭐ **LATERAL is the one
family where the arm gets close to `ha`** — it wins heading and ties cross-track against the
hold-action control — and it is also the only family where the deployment column is essentially
unchanged from the oracle-nav column. The yaw-rate loss is the same "not smooth" signature as the
acceleration loss above.

### 6.5 TACTICAL

Anchor accuracy **0.5654** (chance 1/128 = 0.007812), **50 distinct anchors selected** of 128,
modal anchor `#57` at **0.1482**, selection entropy **2.8425** nats of 4.8520 → **NOT degenerate,
so the read is valid** (the selection void gate, not the trivial gate, is what can see this).

| paired metric | `os − ha0` | `os_navzero − ha0` | ⛔ `os − ha` |
|---|---|---|---|
| lateral manoeuvre agreement | +0.0881 [+0.0580, +0.1233] **WON** | +0.0875 [+0.0576, +0.1212] **WON** | +0.0158 [+0.0017, +0.0273] **WON** |
| longitudinal manoeuvre agreement | −0.0100 [−0.0356, +0.0162] **TIED** | −0.0108 [−0.0367, +0.0156] **TIED** | ⛔ −0.0966 [−0.1189, −0.0739] **LOST** |

⇒ **MIXED vs `ha0` (1 WON / 1 TIED) in both conditions.** ⭐ Against `ha` the split is exactly the
programme's known defect: the arm **wins the lateral decision and loses the longitudinal one**.

### 6.6 STRATEGIC — ⛔ **UNAVAILABLE, n = 0**, quoted from the JSON

```
status:           UNAVAILABLE
n:                0
reason:           "strategic decisions not present in the scored pass (missing
                  ['route_pred', 'route_gt']). A world-model FIDELITY pass does not traverse the
                  hierarchy — run_one prints this explicitly. Producing this family needs a
                  hierarchy-traversing eval, which is a WORK ITEM."
how_to_populate:  "supply `optionset` (map-derived option sets from
                  stack/experiments/nurec-gsplat/strategic_gt.py, consumed by
                  taniteval.strategic_optionset). A route label read off the ego's own future yaw
                  is NOT a substitute: it cannot tell whether the map admitted a choice."
```

`_families_unavailable = ['strategic']` on **every** arm (`os`, `ha`, `ha0`, `os_navshuf`,
`os_navzero`, `oracle_sel`).

⛔⛔ **DO NOT SUBSTITUTE THE ROUTE-HEAD PROBE FOR THIS FAMILY — that conflation is
`RETRACTION_LOG` #16.** The route-head numbers exist and are reported *as a probe*, not as the
family: route accuracy **0.7667 [0.7097, 0.8224]** on 3,622 windows / 128 episodes against a
majority-class no-information rate of 0.6742 — and it reads **the identical 0.7667 under all three
nav conditionings** (true, shuffled, zero), with paired true-minus-shuffled accuracy **exactly
+0.0000 [0.0000, 0.0000]**. ⇒ the head is **nav-INSENSITIVE by construction**, so *"it is not a nav
echo"* is **vacuous** — it cannot echo an input it never receives. It is reading something, but
nothing it reads comes from the route token.

### 6.7 Void gates — the read is VALID

| arm | n | straight | const-speed | trivial (CV plan) |
|---|---|---|---|---|
| `os` | 4,823 | 0.0 | 0.0 | 0.0 |
| `ha` | 4,823 | 0.0402 | 0.0319 | 0.0311 |
| `ha0` | 4,823 | 1.0 | 1.0 | 1.0 (by construction) |
| `os_navshuf` / `os_navzero` / `oracle_sel` | 4,823 | 0.0 | 0.0 | 0.0 |

No arm besides the floor is degenerate, and the selection profile (§6.5) is not degenerate either.
⭐ Both gates are required: a model can select ONE anchor on 100 % of windows while `trivial_frac`
reads 0.0, and the 2-episode probe showed exactly that shape at small n (`n_distinct 8`,
modal 0.3889) before the full read resolved it to 50 anchors at modal 0.1482.

### 6.8 What must NOT be read out of this

* ⛔ **OPEN LOOP, all of it** (PI, 2026-09-02). The rendered report is mechanically guarded against
  the superseded phrase; the guard reported **CLEAN, 0 occurrences**.
* ⛔ **NON-PARITY** — `v2_parity.parity false`, `checked false`, `corpus_key null`. refcv3 is **not**
  cross-arm comparable at the LEVEL with `refc-base` / `refc-xl`. Only each arm's **margin over its
  own `ha0`** travels across the parity break, because `ha0` is bit-identically defined.
* ⛔ **Selection is `out["traj"]` / `sel_score_v3`, never `a_star`.** `oracle_sel` is banked
  separately as **T0** and is a ceiling, never a driveable arm.
* ⚠️ **`os_navzero` is a LOWER BOUND on nav dependence**: with `nav_cmd=None` the E13 injection into
  the tactical and strategic states is skipped entirely, but the **core** collapses onto the
  majority token `follow` rather than losing the input.
* ⚠️ **The tier ruling is OPEN** (BACKLOG R30) — does the doctrine admit as T1 a model consuming no
  actions? Every headline statistic here is a **margin over `ha0` on the same windows**, so the
  ruling changes the row's label, never its arithmetic.

### 6.9 Deliverable manifest

| artifact | repo path | md5 | also at |
|---|---|---|---|
| suite artifact (JSON) | `taniteval/results/refcv3-40284-openloop.json` | `5cfe3258c18871218bd85d691904eb20` | `tanitad-refcv3:/workspace/eval/` |
| suite report (MD) | `taniteval/results/refcv3-40284-openloop.md` | `3ede1c71757deee0791c737c03b4fcdf` | same |
| suite report (HTML) | `taniteval/results/refcv3-40284-openloop.html` | `f6504defb0e1333695b1e8db7fb3b09c` | same |
| arm record — the `ha` contrasts live here | `taniteval/results/refcv3-40284-openloop.ARM.json` | `0de8e8a4ece162332bc3a387fd5d679c` | same |
| per-window dump (141 eps + decisions) | `taniteval/results/refcv3-40284-openloop-dump.tar.gz` | `aff9bad5aff3869c261903d8ff1768e9` | `…/refcv3_40284_dump/` |
| the epoch checkpoint | — | `fc304b62686ddb9e685d14bdab482404` | ⚠️ **POD ONLY** — `tanitad-refcv3:/workspace/experiments/refcv3-b1-v72-30k/ckpt_40284_FINAL.pt` (= `ckpt.pt`), 1,284,991,701 B |

⛔ **The checkpoint lives in exactly ONE place.** The HF push of the final model is the escalation
that removes that; it is the Master Mind's, and it is now unblocked.

