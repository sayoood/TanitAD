# REFe pod handoff — what to provision, what it costs, and what it cannot do

**Written 2026-09-20 before requesting the A40, so the decision is made on measured numbers.**
Everything below is MEASURED on the dev box or PUBLISHED and cited. Nothing is an estimate dressed
as a fact.

## 0 · ⭐ POD RUNBOOK — DATA PREP (added 2026-09-23; read this section first)

Three commands on a fresh Linux pod. Every step is resumable and asserts on its ARTIFACT, not its exit code.

```bash
# 0. ship the refe-plan directory to $WORK/refe-plan (code/ refe/ splits/ -- 17 runtime files, 10 gates, the stage scripts)
# 1. teacher environment, from PUBLIC sources, sha256-verified, CUDA proven by a real conv2d
bash $WORK/refe-plan/code/pod_teacher_env.sh
# 2. data prep: DBs -> camera rigs -> pixels -> rank-0 targets -> augmentation -> scorer -> gates
bash $WORK/refe-plan/code/pod_dataprep.sh     # TAU defaults to 0.3 (PI); STAGES="..." for a subset
```

| stage | what | MEASURED size / cost |
|---|---|---|
| teacher env | DriveZero @ `2495954` (checkpoint git-tracked), Python 3.11, torch 2.7.1+cu128, 113-pkg lock `--no-deps` | 52 MB + maps 0.90 GiB |
| `dbs` | 1,192 navtrain DBs range-fetched from train+val zips | **97.57 GiB transit -> 163.82 GiB disk** |
| `calib` | each log's own camera rig from its DB (R22) | seconds; 26-30 rigs/camera MEASURED |
| `pix` | OpenScene `navtrain_current_{1..32}`, 4 REFe cameras kept | 304.5 GB transit -> **~84 GB disk** |
| `rank0` | teacher rollouts, 4.0 s @ 5 Hz, self-healing shards | dev box **31.7 rows/min @ 4 shards = 7.6 s/rollout/shard** (MEASURED over 50 min / 1,575 rows; the older 17.9 was a short post-restart window); re-measure on the pod |
| `aug` | K=2 diversity search | **3.25 rollouts/frame, 52 % yield at TAU 0.5** (71 % at 0.3) |
| `score` | PDM targets, rank 0 + augmented (per-row route, R18) | |
| `gate` | consumer conformance + signal consistency | must read CONSUMERS_CONFORM / SIGNALS_CONSISTENT* |

⭐ **PI DECISIONS 2026-09-23:** backbone **ViT-L** (the only published DINOv3 anchor, 94.55 PDMS, Table A13) · augmentation **TAU = 0.3 m** (71 % yield) · **ONE pod** for data prep, which then trains.
✅ **Every code path now exercised:** `fetch_navtrain_dbs.fetch_one` fetched a real 2.08 MB DB from `val.zip`; size + CRC asserted inside, and its sha256 is **IDENTICAL** to our independently extracted copy (1,380 `lidar_pc` rows in both).

## 0b · ⭐ POD RUNBOOK — TRAINING (added 2026-09-23, after data prep prints `ZZDONE`)

```bash
nohup setsid bash $WORK/refe-plan/code/pod_train.sh > $WORK/pod_train.out 2>&1 < /dev/null &
# status:  tail -2 $RUN/metrics.jsonl ; cat $RUN/restarts.log ; ls $RUN/snap_epoch*.pt
# after ANY pod restart: the SAME command -- every stage is idempotent, training resumes
```

| stage | asserts, on the ARTIFACT | why it exists |
|---|---|---|
| `weights` | DINOv3 ViT-L + ViT-S from timm's **ungated** mirror at pinned commits; sha256 **==** the LFS hash HF itself reports (`45172f20…`, 1,212,347,640 B) | the trainer refuses a partial trunk (R15); a wrong file must never get that far |
| `preflight` | 4 bank files non-empty, no stray `targets_*.jsonl` (R19), `calib_table.json` + `diag_calib.py` `CALIB_OK` (R22), 32/32 pixel shards, container RAM ≥ 24 GiB, a real 4 GiB `dd` where checkpoints live | never `df` on a pod; RAM is MEASURED ~7.8 GiB of banks |
| `resume` | `diag_train_resume.py` prints `RESUME_EXACT_AND_GUARDS_LIVE` | the run is only as safe as its resume (R20) |
| `batch` | largest RESIDENT batch in powers of two -> `accum = 256 / batch` exactly | WTA coverage needs the paper's effective batch 256 |
| `smoke` | one real step on the FULL bank: `images: N/N tuples resolve ALL cameras`, `calib: N/N tuples carry their own rig`, `TRAIN_DONE` | the pixel join, the decode and the per-sample rigs, on the pod's own disk (R21, R22) |
| `train` | self-healing supervisor; stops on `summary.json {"done": true}`; a deterministic refusal (exit 3/4) is NOT restarted | ~65 days must survive restarts without a human |

**The run directory** (`$DATA/refe_runs/vitl16_navtrain_aug_tau0.3/`): `config.json` (argv, bank + trunk identity, source sha256s) · `metrics.jsonl` (every optimiser step + resume/ckpt/epoch events) · `ckpt_last.pt` (~0.23 GB, every 20 min) · `snap_epochNNN.pt` (~76 MB each, evaluable through `planner.py`) · `model_final.pt` (full, ~1.3 GB) · `summary.json` LAST.

**Timeline, with its evidence class.** ⭐ **An epoch is one pass over SCENES** (PI 2026-09-23, R24 -- the paper's unit: Table A12 counts its data in scenes). **103,039** scenes (103,288 split tokens x the 99.76 % local yield), each visited ONCE per epoch with one of its targets -- the logged intent or, for the ~71 % that have one, its goal-augmented twin, taking turns across epochs (`train.py --epoch-unit scenes`, the default). x 25 epochs = **2.58 M sample-passes**. *(An epoch over the ~176 K stored targets would have been 4.40 M -- 1.71x the paper's budget, ~64-70 days.)* Per sample on an A40: 3.217–3.372 s MEASURED on the 4060 / 2.475–2.559 (A40 multiplier, ESTIMATED) = 1.26–1.36 s -> **~900–975 A40-h ≈ 37–41 days**. ⭐ The first hour of `metrics.jsonl` (`samples` / `elapsed_s`) replaces the estimated multiplier with a measurement.

⚠️ **Data prep is CPU-bound, so the POD'S vCPU COUNT sets its duration.** Rollouts alone: (103,039 rank-0 + 2.70 x 103,039 search) = **381,244 rollouts x 7.6 s** (MEASURED per shard on the dev box over 50 min, 6 threads/shard; 13.4 s in an earlier short window) = **~805 shard-hours** (up to ~1,419 at the slow rate), **plus the scorer: 6.2 s/frame MEASURED** (fake-pod rehearsal, 6.7 rank 0 / 5.6 augmented, one contended shard) x ~176 K frames = **~303 shard-hours** -> **~1,110 shard-hours in all**. At `SHARDS = vCPU / 6`: **16 vCPU -> ~23 d · 32 -> ~9 d · 48 -> ~5.8 d · 64 -> ~4.6 d**. Choose the A40 host by its vCPUs.

**Ship:** the package's `code/ refe/ splits/` as in §0. The subset the pod RUNS is re-derived in §5 from the import graph (`raw/2026-09-23-shiplist/`): **17 runtime** (+ `refe/ckpt_io.py`, `refe/calib_table.py`) and **10 gates** (+ `refe/diag_train_resume.py`, `refe/diag_calib.py`), plus the shell-invoked stages no import graph can see — `code/pod_teacher_env.sh`, `code/pod_dataprep.sh`, **`code/pod_train.sh`**, `code/fetch_navtrain_dbs.py`, `code/requirements_teacher_lock.txt`, `splits/navtrain.yaml`.

## 1 · The number that should drive the decision

⛔⛔ **EVERY HOUR FIGURE HERE WAS WRONG BY ~4× UNTIL 2026-09-21, IN THE DIRECTION THAT MAKES A
PURCHASE LOOK AFFORDABLE.** The `0.780 s/sample` basis was a **ONE-CAMERA** measurement carrying a
**FOUR-CAMERA** model, and the cross-check that made it look trustworthy — *"608.5 A40-hours against
the paper's 608"* — compared **our 1-camera cost against their 4-camera cost**. Two errors of the
same size in opposite places read as agreement. *(Class: a true measurement quoted outside its
scope, which is the programme's most repeated failure.)*

⭐ **The full measured answer is `TRAINING_TIME.md`.** Its basis: the camera scaling is **linear and
MEASURED** — ×3.874 at ViT-S, ×4.087 at ViT-B against an analytic ×4.00 — so the ViT-L 4-camera cost
extrapolates along a *verified law* instead of an assumption.

| | |
|---|---|
| their training (PUBLISHED, Table A12 p. 30) | 100 K navtrain + 237 K SimScale = **337 K samples**, 25 epochs, batch 256, **16 × H20 · 38 h = 608 GPU-hours**, precision **FP32** |
| that is | **8,425,000** sample-passes |
| REFe MEASURED on this 4060: ViT-L, 4 cameras, 512×960, batch 1 | **3.217–3.372 s/sample** |

| arm | backbone | scope | **A40-hours** | **days on ONE A40** |
|---|---|---|---|---|
| **D · full, incl. the augmented SimScale half** | ViT-L | 337 K × 25 | **2,942–3,188** | **123–133** |
| C · navtrain only | ViT-L | 100 K × 25 | 873–946 | 36–39 |
| **B · full, incl. SimScale** | **ViT-B** | 337 K × 25 | **1,042–1,077** | **43–45** |
| **A · navtrain only** | **ViT-B** | 100 K × 25 | **309–320** | **13** |
| A− · navtrain only | ViT-S | 100 K × 25 | 134–138 | 6 |

⛔ **Do not provision one A40 expecting a full reproduction — that is 123–133 days.** *(published as 99–128 until the sixth review)* On 8 × A40
arm D is 15–17 days; scaling is near-linear because the paper's own batch is already split 16 ways.

⭐ **Recommended: arm A (ViT-B, navtrain only, 13 days).** Its 4-camera step time is **MEASURED,
not extrapolated**, and it reproduces a row the paper publishes: Table A13 (§B.2) trains ViT-S,
ViT-B and ViT-L on navtrain **only, without any SimScale**, all other settings fixed. That is the
correct anchor, not a scoped-down improvisation.

⚠️ **The one ESTIMATED term is the A40/4060 multiplier, **2.475–2.559×** (the 3.0 was supported by NONE of the three sources)**, from three published ratios
agreeing within 3 % (FP32 2.48×, bandwidth 2.56×, TF32 2.48×). ⇒ **First job on the pod:
`raw/2026-09-21-camera-scaling/cam_scaling.py`. Three minutes, and the last estimate is gone.**

⚠️ **A cross-check that does NOT close, stated rather than buried:** their 608 H20-hours convert to
≈2,980 4060-hours; our basis implies 7,529–7,891. **REFe is ~4.3–5.3× slower per sample than
DriveZero's implementation** — a cost gap, not a correctness gap. Candidates: no fused attention,
no `channels_last`, batch 1. `TRAINING_TIME.md` §5.

## 2 · The older rehearsal arms

Still useful for *mechanics*, not for driving numbers. ⚠️ Their hour figures carried the same
one-camera basis as §1 did; multiply by ~4.

| arm | samples | epochs | what it buys | what it does NOT buy |
|---|---|---|---|---|
| **rehearsal** | our 2,182-tuple **4-camera** bank | 50 | the loop runs at scale on real data; timings become real | nothing about driving — 8 scenarios |
| **mini-scale** | all 64 mini logs, K=2 routes, ~14 K tuples | 25 | a first REFe that can be SCORED closed-loop against the teacher's 97.19 | not comparable to their table |

## 3 · What is already done, so the pod does not debug

| piece | state | artifact |
|---|---|---|
| architecture, ViT-L, **4 cameras** | ✅ **322,070,854 / 18,991,426 trainable (5.90 %)** vs paper 338.46 M / 18.58 M (5.49 %) — ⚠️ the 316.9 M / 11.88 M this row carried was the ONE-camera, pre-`reg_mlp_ratio` figure | `refe/model.py` |
| fidelity cross-check | ⛔ **INVERTED — see below** | `MODULE_SIZING_STUDY.md` §3.2 |
| real DINOv3 weights | ✅ all 3 sizes load strictly, 0 unconsumed tensors | `raw/refe_dinov3_load_all.txt` |
| training loop | ✅ overfit **1.9316 → 0.3730 (80.7 %)** at four cameras with gradient accumulation; `TRAIN_CONTROL_OK` | `raw/refe_overfit.txt` |
| **real camera frames** | ✅ learning on actual JPEGs | `raw/refe_overfit_realimages.txt` |
| target bank | ✅ **2,182 tuples** (1,091 × 2 genuinely distinct ranks), **2,182/2,182 = 100 % resolve all four cameras and decode** | `data/refe_targets_4cam` |
| goal augmentation | ✅ route → goal → rollout, controls exact | `STAGE1_GOAL_AUGMENTATION.md` |
| planner wrapper | ✅ runs in THEIR harness, so REFe is scoreable | `refe/planner.py` |

## 4 · What to provision

* **1 × A40 (48 GB)** for arms A and B. ⛔ **At FOUR cameras ViT-L peaks at 9.49 GB for batch 1** (MEASURED 2026-09-21). ⚠️ The "9.58 GB for batch 2" this bullet used to carry was the **ONE-camera** rig, and it is the same scope error that made the whole cost table wrong.
  ⇒ the shipping config does **not** fit the dev box's 8 GiB card at ANY batch size, which is
  why every timing here is extrapolated and why the pod must re-measure. 48 GB should hold
  ~batch 4; `pod_bootstrap.sh` §3 reports the largest **RESIDENT** batch rather than the
  largest that merely runs — a batch that spills to host RAM runs and is ~34x slower.
* ⛔⛔ **BLOCKER FOUND 2026-09-22 — THE CAMERA FETCHER CANNOT READ THE SPLIT WE NEED.**
  `fetch_front_camera.py` range-fetches frames by reading a **ZIP central directory**. MEASURED
  from each archive's first 288 and last 256 bytes: **only `mini` is a ZIP**; `test`, `val` and
  **`train` — which holds navtrain's pixels — are TAR archives named `.zip`** (`ustar` at offset
  257, tar zero-block tail, no central directory). train_set is **43 archives (0-42) of ~116 GiB
  = ~4.87 TiB**. The tool was only ever MEASURED on mini, and the `~73 GB test / ~99 GB val`
  figures below were **projected from mini, never indexed** — they could not have been.
  ⚠️ This blocks the TRAINING frame source only; target generation reads the DB `image` table
  (metadata) and needs no pixels.
  ⛔ **The obvious recovery does NOT work and was refuted the same hour.** A binary search over
  headers needs the archive sorted by path; MEASURED at 9 offsets across `train_camera_0`, the
  logs are **interleaved arbitrarily** (5 of 9 probes out of order). Members are grouped by
  `<log>/<camera>/` and carry **8 camera channels**, but there is no global order to search.
  Indexing a tar means walking EVERY header: ~580 k per archive x 43 archives, or downloading
  the 4.87 TiB payload. **The pixel source is therefore OPEN.** Evaluate, in order: (a) the
  OpenScene navtrain set already sized at 449 GB (NAVSIM-native); (b) a published manifest for
  the nuPlan tars; (c) a one-off full index built on a fat pipe and cached. See
  `RETRACTION_LOG.md` R14 + its amendment.
  ⭐⭐ **RESOLVED: use OpenScene `navsim/navtrain_current` (32 tgz, 304.5 GB, ~0.8 h on a pod)**
  instead of the nuPlan tars. Its `data_path` is already `<log>/<CAM>/<token>.jpg` — the exact
  form REFe's rows carry — and `index_images` already indexes loose jpgs as well as containers.
  Download, extract, point `--images-root` at it. ⚠⚠ ONE assumption left: that a shard extracts
  to that tree at its root. ⭐ **CONFIRMED 2026-09-22 from 6 MiB of the archive head** (gzip is a
  stream): the layout is `<shard>/<log>/<CAM>/<token>.jpg` — the expected tree plus ONE extra
  leading directory per shard. **Harmless, no code change**: `index_images` and `FrameStore.read`
  both key on `os.path.basename`, so depth does not matter; extract all 32 shards under one root
  and point `--images-root` there. Frames ~200-240 KB. ⇒ **the pixel chain is closed end to end.**
⭐⭐ **SHARD DOWNLOADED AND DISSECTED 2026-09-22 (PI authorised).** `navtrain_current_25.tgz`,
**4,496,159,137 bytes, byte-exact, HTTP 200**, listed and streamed cleanly (**14,034 members**).

**Layout CONFIRMED at scale:** every jpg path is 4 components, `<shard>/<log>/<CAM>/<token>.jpg`;
38 logs; **8 camera channels with exactly 1,517 frames each** (12,136 jpgs), REFe's four present.
Frame arithmetic closes: this shard is 1.4 % of navtrain_current's 304.5 GB and holds 1.5 % of the
split's 103,288 tokens, so navtrain_current does cover the whole split.

⭐ **MEASURED CONTENT SPLIT — REFe NEEDS ONLY 27.5 % OF WHAT IT MUST DOWNLOAD:**

| content | size | share |
|---|---|---|
| **REFe's 4 cameras (F0/B0/L0/R0)** | 1.22 GiB | **27.5 %** |
| other 4 cameras (L1/L2/R1/R2) | 1.28 GiB | 28.9 % |
| **LiDAR `MergedPointCloud/*.pcd`** | 1.94 GiB | **43.8 %** |
| total uncompressed | 4.43 GiB | |

⛔ **Network cost is unavoidable (304.5 GB): a `.tgz` is a gzip STREAM, so nothing can be skipped
in transit.** But disk is not: extract with a filter and keep ~84 GB instead of ~305 GB.

    tar -xzf navtrain_current_N.tgz --wildcards '*/CAM_F0/*' '*/CAM_B0/*' '*/CAM_L0/*' '*/CAM_R0/*'

⭐ **`navtrain_history` (144.5 GB) is NOT needed at all** — REFe consumes the current frame only.
⭐ **Download rate MEASURED at 7.76 MB/s** on this box (4.19 GiB in 579.7 s), well above the
1.6-3.8 MB/s the 2026-09-20 sizing assumed ⇒ 304.5 GB is ~10.9 h here, not the 22-52 h banked. On a
pod at 100 MB/s it remains ~0.8 h.
⚠️ Not yet exercised: resolving an ACTUAL banked row against these files. Shard 25's 38 logs do not
intersect the 11 logs the bank has reached so far (877 of 18,179 rows). Re-check once the bank
covers a shard-25 log, or fetch the shard holding a log we already have.


* **Disk: ~250 GB — but ONLY if frames are stored in CONTAINERS.** Front-camera-only nuPlan is
  **50.7 GB mini / ~73 GB test / ~99 GB val** of DATA (MEASURED by indexing all 9 mini archives).
  The lidar halves are **never needed** and must not be pulled.
  ⛔ **As LOOSE JPEGs that same data costs 5x the disk.** MEASURED: D: is exFAT with a **1 MiB
  allocation unit**, so a 204 KB frame occupies a whole megabyte — 26,060 frames = **5.08 GB of data
  but 25 GB on disk**. Projected over all three splits that is **223 GB of data costing 1,158 GB**,
  which does not fit beside the 481 GB database pull.
  ⭐ **Fixed, and verified:** `fetch_front_camera.py --container` writes one zip per log and the
  dataloader reads from it. MEASURED round-trip: 4,020 frames, 0.75 GB on disk against 3.93 GB
  loose — **5.3x saved**, all frames decoding at 1920x1080. Check the pod volume's allocation unit
  before assuming this is a D:-only problem.
* **No gated downloads.** DINOv3 comes from the ungated timm mirror; `facebook/dinov3-*` returns 401.

## 5 · What to ship, and what to rebuild there

⚠️ **"`refe/` (7 files)" WAS STALE — there are 30 `.py` files there now.** A ship list that
under-counts is how a pod ends up debugging an import at 2 a.m., so here it is explicitly.

**Ship — the 15 files the pod actually runs.** ⛔ **COMPUTED FROM THE IMPORT GRAPH 2026-09-23,
not maintained by hand** — the hand list below this line read 9 and was missing **6** reachable
files. Derived by an AST walk from the entry points (`train.py`, `build_teacher_rollouts.py`,
`augment_search.py`, `build_scorer_targets.py`, `planner.py`, `validate_model.py`):

`refe/`: `model.py` · `train.py` · **`load_dinov3.py`** · `build_targets.py` · **`build_teacher_rollouts.py`** ·
**`navtrain_scenarios.py`** · **`augment_search.py`** · **`route_rank.py`** · `build_scorer_targets.py` ·
`score_proposals.py` · **`scorer_gate.py`** · `planner.py` · `validate_model.py` · **`ckpt_io.py`** · **`calib_table.py`**
*(re-derived 2026-09-23 after R20 + R22: 15 -> **17**; `train.py` and `planner.py` import both)*
`code/`: `augment_routes.py` · **`route_lane_rank_patch.py`**
⚠️ **Shell-invoked stages are INVISIBLE to an import graph** and ship by name: `code/pod_teacher_env.sh`,
`code/pod_dataprep.sh`, `code/pod_train.sh`, `code/fetch_navtrain_dbs.py`,
`code/requirements_teacher_lock.txt`, `splits/navtrain.yaml`.

⚠️ **`load_dinov3.py` is listed explicitly because the graph did NOT reach it from `train.py` until
2026-09-23** — its sole importer was `diag_rope.py`, which is exactly R15: the trainer never loaded
the trunk. It is now imported by `train.py` and the trainer REFUSES to run on a zero or incomplete
mapping. **Re-derive this list after any edit rather than editing it.**

~~Superseded hand list: `model.py` · `load_dinov3.py` · `train.py` · `build_targets.py` ·
`build_scorer_targets.py` · `score_proposals.py` · `planner.py` · `validate_model.py` ·
`augment_routes.py`.~~

**Ship — the 7 pre-launch gates, because a pod that cannot check itself will launch on a defect:**
`diag_consumer_conformance.py` · `diag_rank_distinctness.py` · `diag_architecture.py` ·
`diag_rope.py` · `diag_schedule.py` · `diag_guard_audit.py` · `diag_planner_holds.py` ·
**`diag_signal_consistency.py`** *(added 2026-09-22; G1/G2 report N/A, not PASS, on navtrain)* ·
**`diag_train_resume.py`** *(added 2026-09-23, R20: halt + resume must be bit-identical; `pod_train.sh` refuses to launch without its PASS)* ·
**`diag_calib.py`** *(added 2026-09-23, R22: per-sample rigs against an independent OpenCV reference; also a `pod_train.sh` precondition)*.
⛔ **`diag_consumer_conformance.py` is the load-bearing one** — it is the only check that sees the
model, the bank, the trainer and the planner disagreeing, which is the failure that cost a whole
evening here while every other guard stayed green.

**Ship also:** `code/pod_bootstrap.sh` (preflight + re-timing + the camera pull),
`code/fetch_front_camera.py`, `raw/2026-09-21-camera-scaling/cam_scaling.py`, and the DINOv3
weights (84 MB / 328 MB / 1.2 GB).

**Do NOT ship:** the four `_probe_*.py` scratch files and the one-off `diag_aimed/batch/cameras/
cost/curbdist/determinism/enrich_cost/goal/keys.py` probes — they are banked evidence for questions
already answered, not runtime code.

**Rebuild on the pod:** the target bank and the scorer bank — both derive from teacher rollouts,
and regenerating is far cheaper than moving them, since the dev-box uplink is **1.2 MB/s**
(MEASURED 2026-09-19). ⛔ **When you rebuild, the rank-1 arm MUST point at the rank-1 rollout
directory** (`m-nr-r1`, not `m-nr-n`): `--rank` only STAMPS a label, and building both ranks from
one directory yields a relabelled copy that doubles every count while adding nothing. MEASURED here
2026-09-21 — `diag_rank_distinctness.py` exists to catch exactly that and must be run after any
rebuild.

## 6 · ⛔ Known gaps the pod will hit

1. ✅ **RESOLVED 2026-09-20 — the scorer works and a target bank is being built.** This entry
   previously said the scorer was INERT and told the pod to budget "the engine-side derivation as
   REAL work". ⛔ **That was wrong in every part and is retracted** (`RETRACTION_LOG.md`,
   `raw/refe_scorer_discriminates.txt`). MEASURED: on the teacher's own path `OffRoad.info` reads
   **0.0000**; on a path aimed through the nearest curb it reads **1.0000** with reward
   **−1.6994**; **10 of 28** signals separate under step-wise scoring. Nothing had to be
   reproduced — the builder's `ScenarioData` was sufficient all along.
   **The three real defects were mine:** (a) my reader collapsed each calculator's nested dict to
   its `reward` key and discarded the `info` channel that carries the signal; (b) I scored only the
   trajectory's ENDPOINT, while these are per-step crossing detectors, so a curb crossed at step 6
   read clean; (c) my "25 m sideways veer" control moved the ego **away** from the nearest curb
   (10.66 m → 19.06 m) and never violated anything.
   ⛔ **SUPERSEDED 2026-09-22 — this said the targets are "not yet wired into `train.py`", and
   they now are.** VERIFIED in `refe/train.py`: `ScorerBank(a.scorer_targets)` (l.407) is passed
   into `TargetBank(scorer=...)` (l.426), joined per frame on `(log_name, token, int(step),
   int(rank))` (l.274-275), and the score term is in the objective — `loss = (l_traj + l_score)
   / a.accum` (l.563), normalised by the bank-derived constant `cov_norm` (l.487-492). A frame
   with no scorer targets contributes an all-zero mask and NOTHING to the score loss (l.337-346),
   so an aborted frame degrades the signal rather than crashing the run.
   ⭐ **RESOLVED the same day — see `RETRACTION_LOG.md` R13.** The pilot aborted **8 of 18**
   navtrain frames against **ZERO** on val14, and the split counter showed **8/8 were `!ok_curb`
   and 0/8 were `ndiff < 3`**, with those frames carrying **19-20 separating signals**. The abort
   message ("scorer could not tell candidates apart") was false for every one. `ok_curb` demanded
   that the analytic over-curb candidate actually read off-road — a property of ROAD GEOMETRY,
   which navtrain's wider roads defeat and val14's curated urban scenes never did. It is now a
   DIAGNOSTIC, not a gate: navtrain recovers **18/18 frames, 198 rows, 0 aborts** (`S1 18/18`),
   and val14 is content-neutral (**22 rows identical, 0 DIFFERENT**, over-curb diagnostic **0**). Bank cost MEASURED at
   **~12 s/frame** (9 candidates, stride 2, single core; `ScenarioData` build 2.5–4.9 s dominates),
   so a 982-frame rank is ~3.3 h single-process and ~50 min across four.
   ⭐ **All six components discriminate — MEASURED over 664 banked rows, none is constant.**
   `goal_reaching` orders correctly: `lon x1.5` reaches a goal **45/74**, `teacher` **13/74**,
   `lat±2` **10/74**, `stopped` **0/73**. ⚠️ An earlier line here called it unvalidated on the
   strength of **one frame** where no candidate happened to reach a goal; that is retracted. It
   fires when a step's swept segment passes within the threshold, so a frame whose goals sit ~12 s
   ahead reading 0 everywhere is correct behaviour.
2. **8 scenarios is not a training set.** Arm A cannot generalise and must not be quoted as if it can.
3. **The A40/4060 multiplier is unmeasured.** Replace it with a real timing before committing to C.
4. ⭐ **BATCH SIZE IS NOT A THROUGHPUT KNOB HERE -- it decides whether the 64 proposals exist.**
   WTA gives gradient to exactly one proposal per SAMPLE, so at batch 2 at most 2 of 64 proposals
   are touched per step. MEASURED: at batch 8 winner diversity ran 1-5 per step, at batch 2 it was
   **1/2 on every logged step** -- both samples choosing the same proposal. **Their batch was 256.**
   ⇒ the dev box cannot train this head properly at any batch it can hold, which upgrades the pod
   from *faster* to *necessary*. The first pod job should measure the largest batch that fits on
   48 GB and report it.
   ⚠️ That the starvation CAUSES the real-image plateau is a hypothesis, not a finding: that run
   changed images, backbone and batch at once. The isolating arm is queued
   (`raw/refe_wta_batch_starvation.txt`).
5. **Batch 4 on an 8 GB card silently spills to host RAM** rather than raising OOM. On the pod, size
   the batch by measured peak against the card, never by "it ran".


### ⛔ CORRECTIONS 2026-09-20, from the third review and the module sizing study

**The arm table above was inconsistent by exactly 9.0×.** Arms A and B were quoted at 0.7 and 2.8
A40-hours; recomputing them on the same basis as arm D (which checks out at 608.5 against a quoted
608) gives **6.3** and **25.3**. The sentence *"the first pod day costs under three GPU-hours"* was
the basis of the A40 request and is **withdrawn**: the first pod day is ~6.3 A40-hours for arm A
alone.

**The parameter fidelity cross-check has inverted and is no longer evidence of anything.** Three
documents said our 4-camera reconstruction lands "within 0.03 percentage points" of the published
5.49 %. MEASURED after the register-compression module was added: **9.693 %**, i.e. trainable
**+75.1 %** over their 18.58 M. The PI accepted the parameter cost of compressing at width 1024;
the PI was not told that accepting it voids the package's only independent fidelity check.
⇒ Following the sizing study, REFe now stands at **18,934,338 trainable / 322,013,762 total
(5.88 %)** against their **18.58 M / 338.46 M (5.49 %)** — **+1.9 % on trainable, with ONE camera
against their four.** The cross-check is restored, but by CHANGING THE MODEL, not by re-deriving.

**Peak memory is 9.58 GB at batch 2 ON ONE CAMERA**, not 5.36. ⚠️ SUPERSEDED 2026-09-21: the rig is now FOUR cameras and peaks at **9.49 GB at batch 1**, so the shipping config does not fit this card at any batch size. `validate_model.py` prints it and never asserts
it. It fits an A40 with room; it does **not** fit the dev box's 8 GiB RTX 4060, so any local
rehearsal must drop to batch 1 or a smaller backbone.


---

## 7 · The navtrain target path — ADDED 2026-09-22, and it is the critical path

⭐ **Why this section exists.** Every arm in §1 trains on **navtrain**, and until today the target
builder could not reach it. `build_teacher_rollouts.py` read its scenarios out of CLOSED-LOOP SIM
LOGS (`--run <dir>/**/*.msgpack.xz`) — the artifact their val14 harness leaves behind. **navtrain has
no such runs.** The split is a list of `(log, token)` pairs, so the scenarios must be constructed
from the DB. That is now `refe/navtrain_scenarios.py` plus `--source navtrain`.

### 7.1 · What navtrain is, MEASURED

| | |
|---|---|
| split file | `…/navsim/devkit/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml` |
| size | **1,192 log_names · 103,288 tokens** — matches the paper's "100K navtrain" |
| frame spec | `num_history_frames 4 · num_future_frames 10 · frame_interval 1 · has_route True` |
| nuPlan DBs we hold for those logs | **214 / 1,192 (18.0 %)**, all in `trainval` |
| navtrain frames reachable locally | **18,179 / 103,288 (17.60 %)** |

⚠️ The token list is FLAT and carries no log label, and `len(tokens) != len(log_names)`, so the
pairing must be read out of each DB's `lidar_pc` table. Do not infer it positionally.

### 7.2 · Cost, MEASURED twice by independent routes

| term | value | basis |
|---|---|---|
| **teacher rollout, 40 steps** | **5.94 s** | two-point solve on val14: L2 = 20 rollouts/158.4 s, L6 = 60/396.0 s ⇒ slope 5.94 s, intercept 3.96 s/log. The fit reproduces L6 **exactly** |
| same, on navtrain scenarios | **5.84 / 5.92 / 5.95 s** | direct timing, steady state (first rollout 10.13 s is warm-up) |
| `build_controller` per frame | **0.013 s** | per-token scenarios are affordable: 0.3 % of a rollout |
| `planner.initialize` per frame | **0.005 s** warm (0.629 s cold, once per map) | |
| `NuPlanScenario` construction | **0.002 s** | |

⇒ **local 18,179-frame subset = 30.0 h single-process. Full 103,288-frame navtrain = 170.4 h
single-process.**

### ⛔⛔ PARALLELISM: MORE SHARDS MADE IT SLOWER, AND THE FIX IS ONE ENV VAR

MEASURED on this 24-core box, identical 48-frame workload:

| shards | wall | speedup |
|---|---|---|
| 1 | 323 s | 1.00x |
| 2 | (24-frame run) | **1.93x** |
| **4** | **199 s** | **1.62x — WORSE than 2** |
| 8 | killed at 9/48 after 548 s | **collapse** |

⭐ **Cause: torch's default all-core intra-op parallelism.** MEASURED, ONE shard runs **66
threads**, so 8 shards put **528 threads on 24 cores**; they burned ~19 cores for 8 minutes and
produced **zero rows**. ⇒ pin `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` to
about `cores / shards` before launching. A shard count chosen without pinning is a slowdown.
⚠️ The 1.93x and the 1.62x come from DIFFERENT probe runs, and the second overlapped a foreign
`pytest stack/` consuming ~1 core for its whole duration — so treat the ordering (2 > 4 > 8) as
the finding, not the exact ratios. **Re-measure on the pod; the curve is machine-specific.**
⛔ CORRECTED 2026-09-23: the "9 % GPU" figure was sampled DURING the N=8 thread thrash (528 threads on 24 cores), when the CPU starved the GPU -- it was never the workload's own load. With threads PINNED (4 shards x 6), six samples 5 s apart read 5/36/16/73/42/32 % (mean ~34 %, one earlier spike at 97 %): the load is BURSTY -- a GPU policy forward between CPU map/route/controller work. **Both CPU cores and the GPU matter; neither alone is the constraint.** Do not size a pod for this workload on either number alone.

⛔ **Do NOT convert these with the A40/4060 training multiplier (2.475–2.559×).** That multiplier is
for the ViT-L training step. A teacher rollout is ~148 ms/step and looks CPU-bound (map features,
route search), so the pod's lever is **process parallelism, not the card**. The dev box has 24
cores. ⚠️ **Parallel scaling is NOT yet measured** — measure it before planning the pod run, and do
not assume linear.

### 7.3 · What the pod must pull for target generation — and what it does NOT need

| | |
|---|---|
| missing navtrain nuPlan DBs | **978** of 1,192 |
| **MEASURED size** | **88.80 GiB transit -> 148.83 GiB on disk** (exact, from the zip central directories). ⛔ CORRECTED 2026-09-23 by MEASUREMENT: the 978 missing DBs are **88.80 GiB in transit -> 148.83 GiB on disk**, read exactly from the nine train-zip central directories (`code/fetch_navtrain_dbs.py --plan`, 978/978 found). The earlier ~68.5 GiB was **2.2x low**: it was sized from the 214 DBs we hold, which come from nuPlan **val**, whose logs are smaller than train's -- an estimate from an unrepresentative sample. nuPlan train ships ONLY as city zips (947.4 GiB total); these are real ZIPs with a tiny central directory, so each DB is range-fetched individually (10.7x less than the whole zips). |

⭐ **POD-COMPLETE FIGURE (MEASURED 2026-09-23, simulating an empty pod with `--held-root` pointed at an empty dir):** **1,192 / 1,192** navtrain logs found across the 9 train zips **plus `val.zip`**; **97.57 GiB transit -> 163.82 GiB on disk**, against 1,037.7 GiB for the whole zips (10.6x more). ⚠️ The first fetcher searched TRAIN only (it subtracted what the dev box held), so on a pod it reported *"214 wanted logs are in NO train zip"* -- the 214 we hold came from nuPlan **val**. Fixed.
| maps | all four packs already held; the 214 local logs span all four cities (67 BOS / 56 LV / 51 SG / 40 PIT) |

⭐ **Camera PIXELS are NOT needed to generate targets.** `build_targets.camera_index` reads the DB
`image` table, which is **metadata** (3,865,435 rows across the 214 local logs). Only the TRAINING
run opens the `_CAM_*.zip` pixels. ⇒ the **449 GB** OpenScene navtrain pull is on the critical path
for *training*, not for *target generation*, and the two can proceed in parallel.

### 7.4 · Two defects the build of this path produced — both now closed

⛔ **(a) The route is keyed on `initial_lidar_token`.** The cheap whole-log scenario shape (one
scenario per log, tokens as iteration indices) would have handed the teacher the route of the log's
FIRST frame for every frame in it. **Avoided by design — one scenario per token — and the defect arm
was then BUILT rather than asserted.** MEASURED over 1,061 navtrain frames in 3 logs: per-token route
identical to the whole-log route **27 / 1,061 (2.5 %)**; mission-goal displacement **median 646.1 m,
max 1,425.4 m**. ⚠️ Every rollout would still have been self-consistent and every gate green — the
teacher would simply have driven toward a goal half a kilometre away.

⛔ **(b) The devkit's own token query silently drops 25.4 % of frames, with a bias.** See
`RETRACTION_LOG.md` R12. `get_scenarios_from_db` restricts to `valid_scenes` (first 2 and last 2
scenes); a scene here is **~350 `lidar_pc` rows**, not the ~20 I assumed, so the rule costs
`4 / n_scenes` — **worst on the SHORTEST logs**. MEASURED: 13,565 / 18,179 kept, **19 logs yielding
ZERO**. Replaced by REFe's exact requirement (20 rows back, 40 forward) asserted per frame:
**18,179 / 18,179**. The sibling `include_cameras` filter would have cost a further **394** frames
(97.83 % vs 100.00 %) and is likewise off — REFe's own 4-camera ≤60 ms pairing already runs per frame.

### 7.5 · Controls that passed, and what each one rules out

| control | result |
|---|---|
| scenario equivalence vs the devkit, 1,875 shared tokens | **0 disagreements** across ego state (10 scalars), the full 2.0 s history (20 × 10), tracked objects at iteration 40, route, mission goal, map name, scenario type, timestamp, iteration count |
| history/tail margin over all 18,179 frames | min iteration index **30**; **18,179/18,179** have ≥2.0 s history AND a full 4.0 s tail — no clamped buffers |
| val14 byte identity after the shared-`make_row` refactor | **byte-identical** to the pre-refactor bank |
| navtrain end-to-end smoke | 3 rows, 3 distinct `(log, token, step)` keys, 4 cameras each at ~21 ms sync |

⚠️ **The first version of the equivalence control was worthless**: it compared `repr(EgoState)`, and
`EgoState` defines no `__repr__`, so it compared MEMORY ADDRESSES and reported 1,852/1,875
"disagreements" it would have reported however correct the code was. **A control that cannot come
out the other way is not a control.**

### 7.6 · Commands

```
# local validation subset (30.0 h single-process; --resume makes it restartable)
python build_teacher_rollouts.py --source navtrain --out <dir> --rank 0 --resume

# pilot first
python build_teacher_rollouts.py --source navtrain --out <dir> --rank 0 --limit-logs 4
```

⭐ **`--resume` keys on `(log_name, token, step)` — the same key the scorer bank and the consistency
gate join on — and re-derives it from the written rows, so the file IS the progress and the two
cannot disagree. Rows flush per log, so a kill loses at most one log.**

⛔ **CORRECTION to §5.** That section tells the pod *"`--rank` only STAMPS a label"* and *"the rank-1
arm MUST point at the rank-1 rollout directory (`m-nr-r1`)"*. **That is now stale and, for this path,
wrong.** Since R11, `refe/route_rank.py` makes `--rank` **SELECT** the route in-process and refuse
both silent mismatches. The navtrain source has **no run directory at all**, so the rank can only
come from `route_rank`. `main()` applies it before either branch. Keep running
`diag_rank_distinctness.py` after any rebuild regardless — it is what caught R11.
