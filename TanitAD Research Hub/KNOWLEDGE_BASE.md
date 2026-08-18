# KNOWLEDGE_BASE — TanitAD, consolidated

> **One knowledge base.** The seven per-area files were merged here on 2026-08-18 (PI directive);
> each of them is now a stub pointing at this file, so old links still resolve. Every entry keeps
> its originating area as a tag.
>
> **Format:** `- [YYYY-MM-DD] [source] finding — impact: H_x / WP_y — link`
>
> ⛔ **This is the FINDINGS layer, and it is NOT the paper.** `Paper/TANITAD_PAPER.md` is the
> scientific account of the frontier work — derivations, argument, results in narrative form.
> This file is a curated, deduplicated, newest-first *log of what we learned*, written for an
> agent about to make a decision. Neither is a substitute for the other, and neither should be
> edited to look like the other.
>
> ⛔ **The EVIDENCE layer is `Library/`** — every `[PUBLISHED]` entry should cite a library key
> (`kb_add.py`), not only a URL. A URL is a claim about the internet; a banked sha256 is a claim
> about a file we hold.

**170 entries · merged from 7 area files · 5 cross-area duplicate(s) folded**

| source file | entries |
|---|---:|
| `Architecture & Inference/Research/KNOWLEDGE_BASE.md` | 27 |
| `Benchmarks & Eval/Research/KNOWLEDGE_BASE.md` | 33 |
| `Data Engineering/Research/KNOWLEDGE_BASE.md` | 26 |
| `Opponent Analyzer/Research/KNOWLEDGE_BASE.md` | 46 |
| `Production & Optimization/Research/KNOWLEDGE_BASE.md` | 0 |
| `Project Steering/Research/KNOWLEDGE_BASE.md` | 1 |
| `Tools&DevEnv/Research/KNOWLEDGE_BASE.md` | 42 |

---

- `[Architecture & Inference]` [2026-08-18] [PUBLISHED/library] ⭐⭐ **FROZEN ENCODERS SUCCEED IN EXACTLY TWO CONFIGURATIONS, AND
  REF-A WAS IN NEITHER.** (A) huge frozen VLM + wide interface + supervised head — FROST-Drive
  [`2601.03460`]: frozen 14 B **8.17 RFS / ADE@3s 1.04 m** BEATS the *same encoder fine-tuned*
  **8.13 / 1.47**, while a frozen **ImageNet** ViT is the WORST arm in the table **7.39 / 2.28** ⇒
  freezing is a **multiplier on pre-training quality, not an independent good**; interface width is
  its own lever (5120-d 8.17 vs 256-d 7.68). (B) moderate frozen encoder + **future-feature
  prediction** + **test-time planning** — DINO-WM [`2411.04983`] (frozen DINOv2 **patch** features,
  plain latent L2, *no* reconstruction/reward/terminal loss, **no policy head**, CEM+MPC; swapping
  patch features for global R3M/ResNet18/**CLS** "significantly degrades"), V-JEPA 2-AC
  [`2506.09985`] (<62 h of robot data), and in DRIVING: DeepSight [`2605.10564`] (frozen encoder +
  MSE against DINOv3 future BEV features, Bench2Drive DS **86.23**) and LAW [`2406.08481`].
  ⛔ Full fine-tuning is not the alternative — it DEGRADES pretrained structure (OpenVLA 36.7 % →
  12.1 % under paraphrase); the winning form is a **DUAL ENCODER** (frozen anchor ‖ trainable),
  35.03 → 55.55 → **78.46** [`2509.11417`]; CortexBench agrees from the other side [`2303.18240`].
  ⇒ REF-A had A's consumer on B's encoder class. **This is the evidence base for REF-A v1.**
  — impact: H4 / the encoder question / REF-A v1 — `Research/2026-08-18-frozen-encoder-literature/FROZEN_ENCODER_LITERATURE.md`, primaries in `Library/`
- `[Architecture & Inference]` [2026-08-03] [repo/MEASURED] **The LONGITUDINAL family's distance-keeping half is implemented and its
  gauge is ADMITTED.** `four_families.longitudinal` had returned `distance_keeping: UNAVAILABLE` since the
  binding rule landed (2026-08-02) because our ingest never read `obstacle.offline`. Now: `lead_metrics.py`
  (headway / time-gap / min-TTC) + `build_lead_tracks.py` (the rig→world→t0 frame composition, which is the
  genuinely new part — `lead_state_gate.lead_frame` answers "where is the lead NOW", scoring an arm needs
  "where would it have been relative to the path THIS ARM PREDICTED"). **Pre-registered D-LEAD-1 GT-vs-CV
  control PASSED on all three:** Δ min-TTC **+1.7474 s** [1.5813, 1.9218], Δ headway **+0.9769 m**
  [0.8830, 1.0758], Δ time-gap **+0.1641 s** [0.1499, 0.1786]; paired episode-cluster bootstrap, **14,027
  windows / 1,431 clip clusters**, B=2000, all separated with the correct sign. ⛔ Says NOTHING about any
  arm — it measures the gauge. ⛔ min-TTC is **censored** at 30 s on ~50 % of windows; quote `n_closing`.
  ⛔ **The eval path is not yet fed** — arm evals still report UNAVAILABLE until `win["lead"]` is built for
  the 40 val episodes (backlog P0 L1). — impact: the binding four-family rule / H-LONG / D1
  — `Research/2026-08-03-longitudinal-distance-keeping.md` + `…/incoming/2026-08-03-longitudinal-distance-keeping/raw/dlead1_discrimination.json`
- `[Architecture & Inference]` [2026-08-03] [PUBLISHED] **ADE does not predict closed-loop driving score: ρ = −0.36, p = 0.43 (n=8)**
  ([2605.00066](https://arxiv.org/html/2605.00066)); PDMS aggregate ρ = 0.90, Ego Progress alone ρ = 0.83.
  Direction is citable, magnitudes are not (n=8, p-values, no CI). ⇒ an arm ranking resting on ADE alone is,
  on the field's own evidence, uninformative about closed-loop driving — which is the empirical case for
  Sayed's binding four-family rule. — impact: all gates / instrument doctrine — `Research/2026-08-03-sota-scan/SOTA_SCAN.md` §2
- `[Data Engineering]` [2026-08-02] [measured/corpus] **A "clean v2 val" is NOT clean for v1 — 62 of a 600-clip draw sit in v1's
  TRAIN split** (24 in v1's val). Disjointness proofs are written against ONE corpus and are silent about every
  other arm that will be scored on the split; C64 in mirror image. ⇒ any new split excludes **every** corpus an
  arm was trained on (`load(..., exclude_paths=[...])`). Parity-free remainder = **8,298** of 9,987, so 600 clips
  is **not available** once v1 is excluded (headroom 0.95) — **n=400 is the largest fully-clean balanced split**
  (max |d| 0.0409, cell-census 69.4 %, sha256 `abe041db72a045b3…`). Bonus, no pod needed: at CLIP granularity
  **256 of v1's 600 parity-val clips (42.7 %) are inside v2corpus's training selection** (≠ C64's 21/40 EVAL
  EPISODES — different unit, never merge) — impact: C64/A3/A5/MODEL_REGISTRY comparability —
  `2026-08-02-v2-clean-val-semantics-and-selector.md` §5
- `[Data Engineering]` [2026-08-02] [measured/method] **Stratum headroom belongs to its AXIS SET, and cell-matching ≠ balance.**
  The same remainder gives headroom **6.77× (junction only) → 4.05× (+has_turn) → 1.07× (+speed) → 0.77 =
  INFEASIBLE (+has_brake)** at n=600 — so "6.77× ⇒ feasible" was one axis wide, exactly like quoting an exponent
  without its window. And the cell-quota design it justified leaves **max |d| 0.3997 (10/13 axes over the 0.10
  bar)**; greedy covariate balancing on the four-family axes reaches **0.0094** in 0.2 s, the shipped hybrid
  (quota+balance) **0.0532** with cell-L1 0.0047. ⚠️ Neither is exchangeable: within each matched cell the
  remainder is still skewed (median |d| **0.359**, p90 0.915) and max KS stays 0.12–0.19 vs a 0.057 critical
  value — the residue of a quota selector cannot be matched back into its parent, only mean-balanced — impact:
  every future split/selector, D1 strata, four-family reporting — same note §2–4
- `[Data Engineering]` [2026-08-02] [measured/schema] **`v2_pool_scored.parquet` semantics are now a machine-checked contract**
  (`pool_columns.py`, 34/34 checks PASS on 18,988 rows, corruption-tested): `lk,tl,tr,ac,bs(+v2)` are frame
  **COUNTS** out of `nlab` (⇒ `lk_rate` train **0.4500** vs remainder 0.6718, reproducing the design's
  "lane_keep 45.0 %"); `stopped/city/hw/stop_frac` are **FRACTIONS** with `stopped+city+hw ≡ 1`; `nlab ≡ 179`
  for every clip so the `lk` misread was a pure scale error. Also: the pool has **18,988 rows but 18,987 unique
  clips** — `32ad1a3a-…` is registered under chunks 1573 AND 3117 and is in the v2 selection (both figures were
  in circulation, neither labelled) — impact: A3 unblocked, any consumer of the pool — same note §1
- `[Data Engineering]` [2026-08-02] [measured/absence] **PhysicalAI-AV has NO session/drive id and NO absolute clock** — probed at
  four locations: `clip_index` (3 cols), `data_collection` (5 cols), `feature_presence` (36 presence flags), and
  `egomotion.timestamp` itself, which starts at −0.2 ms and spans ~137 s ⇒ **clip-local microseconds**. ⇒ the
  L2D-style time-overlap dedup that retired that trap on 2026-07-22 **cannot be run here**, and with no lat/lon
  either there is no cross-clip identity linkage on any axis we hold. **Clip granularity is the provable ceiling
  for every PhysicalAI split** — state it, don't imply drive-level cleanliness — impact: I3/split doctrine/C64 —
  same note §8
- `[Benchmarks & Eval]` [2026-08-02] [this run / measured] ⭐⭐ **CROSS-TRACK SAYS THE MODEL WINS; CURVATURE SAYS IT IS 3×
  WORSE THAN THE TRIVIAL FLOOR.** Four-family panel (CLAUDE.md binding rule, landed mid-run) applied to
  the arm **and every floor**, n=881: flagship-v1 `cross_mae_m` **0.1152** vs CTRV 0.1604 (+0.0452
  [0.008, 0.084] separated, favours model) but `curvature_mae_1pm` **0.026969** vs CTRV **0.008967**
  (3.0×) and vs the straight-line floors 0.012221 (2.2×). **The path hits roughly the right points with
  the wrong SHAPE** — invisible to ADE and to cross-track. **REF-C-XL's curvature is 2.2× better than
  the flagship's while its ADE is worse (0.4714 vs 0.4271) — ADE inverts the ordering.** Also
  longitudinal: flagship `speed_bias` **+0.1911 m/s (too fast)** vs REF-C +0.0209; `along_final_bias`
  **+0.3375 m** vs floors ≈−0.11 (separated, favours floor) — impact: the "lateral only" narrative,
  arm selection, v2.1 lever — `../Implementation/incoming/2026-08-02-ctrv-floor/raw/four_families_vs_floors.json`
- `[Benchmarks & Eval]` [2026-08-02] [this run / instrument] **TACTICAL and STRATEGIC are UNAVAILABLE on the canonical eval
  surface, and that is a WORK ITEM.** `rollout.collect` is a world-model *fidelity* pass under the
  expert's true future actions (`pc2_pass=False`, `actions_source="expert_future"`), so no manoeuvre or
  route decision is decoded — `maneuver_pred/gt`, `route_pred/gt` are absent. Same for LONGITUDINAL
  distance-keeping (headway/TTC): no lead-agent track is read, though `obstacle.offline` exists on
  97.44 % of the corpus. ⇒ **two of the four binding families cannot be reported from any current eval**
  — impact: H1b hierarchy claim, binding-rule compliance — same intake.
- `[Benchmarks & Eval]` [2026-08-02] [this run / instrument] **A per-window reimplementation of a pooled-mean metric is NOT
  the same statistic.** `four_families` reduces heading/yaw-rate/curvature as a pooled mean over valid
  steps; a per-window form (needed for a paired bootstrap) is a mean-of-per-window-means and differs by
  7.6e-01 / 5.9e-01 / 3.6e-02 on flagship-30k. The driver measures the disagreement per metric and
  **refuses the interval above 1e-3** rather than bootstrap a different statistic than the one
  published. *(Its first cut returned heading in RADIANS and silently dropped curvature and yaw-rate by
  mis-keying `_seq_geometry` — the C63 failure mode; the agreement check is what caught it.)* — impact:
  estimator hygiene / G-B1 — same intake.
- `[Benchmarks & Eval]` [2026-08-02] [this run / measured] ⭐ **THE CANONICAL DRIVING GATE'S FLOOR CANNOT TURN.**
  `taniteval/driving.py:304` is `FLOORS = ("cv", "holdv0")` — both straight lines — while CTRV, same
  information budget, is **already computed on every window by `driving_diagnostic.baseline_waypoints`
  and discarded by `rollout.collect`**. On the canonical 881-window/40-ep val, CTRV is the DOMINANT
  floor: ADE **0.5265** (gated) vs CV 0.8377 / hold-v0 0.7876; paired **+0.3113 [0.167, 0.484]
  separated**; wins **423/881** windows (CV 156, hold-v0 302) — impact: every lateral/turn verdict,
  D1 gate, LEADERBOARD §0/§1b — `../Implementation/incoming/2026-08-02-ctrv-floor/`
- `[Benchmarks & Eval]` [2026-08-02] [this run / measured] **16 of 25 banked arms' headline verdicts move when CTRV is the
  floor.** 12 arms beat the floor under CV; **6** under CTRV, best surviving margin **+0.0890 m**.
  ⭐ **flagship-v1 @30k: vs CV +0.4106 separated → vs CTRV +0.0993 [−0.026, +0.220] NOT separated** ⇒
  "the FIRST arm below EVERY trivial bar" is a **point-estimate** claim, a **tie** under the program's
  own paired estimator. Escalated (PROJECT_STATE/MODEL_REGISTRY not agent-editable) — impact: the
  program's central capability claim — same intake.
- `[Benchmarks & Eval]` [2026-08-02] [this run / measured] **The lateral win is REAL but ~5× smaller than published.**
  Pre-registered criteria all held for flagship-v1: `sustained_turn` ADE +1.8063 → **+0.3398 [0.153,
  0.550] still separated, favours model**; sharp-curvature heading +24.93° → **+7.69°**; overall
  \|cross\| +0.7720 → **+0.1372**. ⇒ `where_the_win_lives = "lateral only"` survives as a DIRECTION and
  may never again be quoted with a CV-derived magnitude — impact: H15 / v4-v5 gate narrative — same intake.
- `[Benchmarks & Eval]` [2026-08-02] [this run / measured] **At the top speed decile CTRV beats the model LATERALLY, not just
  longitudinally** (n=89: CTRV ADE **0.0986** vs model **0.7159**, 7.3×; \|cross\|, heading and
  crosstrack all flip tie→floor; `speed_high` n=294 ADE flips tie→floor −0.2154 [−0.386, −0.030]). The
  high-speed weakness was framed as purely longitudinal because a straight-line floor cannot expose a
  lateral one on a locally-arc road — impact: v2.1/v3 high-speed lever, `memory/flagship-longitudinal-lever` — same intake.
- `[Benchmarks & Eval]` [2026-08-02] [nuScenes devkit / published] **The community physics floor is a best-of-FOUR including
  two yaw-rate models** — `PhysicsOracle` = {const-velocity+yaw, **const-velocity+yaw-rate (CTRV)**,
  const-accel+yaw, **const-accel+yaw-rate**}, best-per-sample vs GT. Ours was a best-of-two with both
  yaw-rate members removed. Also fixes our own labelling: **best-of-N is an ORACLE (privileged), never a
  competitor** — impact: G-B1 leaderboard hygiene / floor design —
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md
- `[Benchmarks & Eval]` [2026-08-02] [this run / measured] **Three banked window dumps use a DIFFERENT `eid` encoding.**
  `windows_flagship-v4.1-10k / v4.2-step4000 / v16-ab-ft` label the same 40 episodes with the packed
  string uid (`808464434`, …) where all others use `0..39`, with **bit-identical** `gt`/`cv`/`speed`.
  Any cross-arm join keyed on `eid` mis-joins these three; a literal-equality alignment check refuses
  three provably-aligned arms (compare the PARTITION instead) — impact: harness hygiene — same intake.
- `[Opponent Analyzer]` [2026-07-24] [Avride/NHTSA] FACT — NHTSA ODI opened an investigation (**2026-05-08**) into **Avride**
  (Uber robotaxi partner, Yandex SDG lineage): **16 crashes + 1 minor injury**, all tied to **"the
  competence of"** the system — lane-changing, same-lane vehicle response, stationary-object response —
  impact: **new opponent; W-08 → H15/A9; SC-13** stationary-object/same-lane spec — https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/
- `[Opponent Analyzer]` [2026-07-24] [Waymo/NHTSA] FACT — the construction-zone recall (26E035) was Waymo's **2nd in ~1 month**;
  it **pulled all robotaxis from highways 2026-05-19**; filing names the mechanism (mis-prioritizing
  hazard-avoidance / not recognizing the work zone). Separately a Waymo **ran a red light in Dallas** amid
  a new federal probe — impact: **W-01 enrich; W-03 family → SC-14** red-light barrier — https://techcrunch.com/2026/06/18/waymo-recalls-nearly-4000-robotaxis-to-stop-them-driving-into-highway-construction-zones/
- `[Opponent Analyzer]` [2026-07-24] [Tesla/NHTSA] FACT — FSD probe **upgraded to Engineering Analysis 2026-03-18**, **~3.2 M
  vehicles**, **9 crashes / 1 fatality / 2 injuries**; the **"degradation-detection" feature** fails to
  flag impaired cameras until immediately pre-crash. Tesla also **unredacted 17 Austin robotaxi ADS
  incidents** (2 with teleoperators); Miami launched into rain 2026-07-03 — impact: **W-04 → H11/H15/H2
  strongest field validation** — https://electrek.co/2026/03/19/nhtsa-upgrades-tesla-fsd-visibility-investigation-3-2-million-vehicles/
- `[Opponent Analyzer]` [2026-07-24] [arXiv] FACT/INFER — **Metis** (2606.15869, Fudan/HKU/Tongji/Li Auto, subm. 06-14):
  "efficient world-action model" — Mixture-of-Transformers with separate video-gen + action experts and
  an **asymmetric attention mask that lets the action head skip generative rollout at inference** (its
  efficiency lever ≈ our latent path). SOTA NAVSIM navhard/navtest + CityWalker. **But no hierarchy, no
  in-loop imagination, no self-monitoring, and NO param count / compute-normalized metric** → not a true
  CNCE competitor — impact: sharpens H1/H3/H5/H15/H11 wedge; publish params+CNCE it doesn't — https://arxiv.org/abs/2606.15869
- `[Opponent Analyzer]` [2026-07-24] [Momenta/Autobrains/NVIDIA] FACT — **Momenta listed HK 2026-07-08** (~$8.9 B cap; Mercedes+
  BYD cornerstones); shipped **R7 RL World Model (Apr'26)** + **X7** chip (SAIC-VW ID.ERA 9X). **Uber's
  Munich robotaxi pilot shifted to Autobrains+NVIDIA (2026-06-02)** → Autobrains stepping ADAS→L4. **NVIDIA
  Mercedes-Benz CLA** ships the full Alpamayo stack (US, this quarter); family **10 B (Nano) → 32 B
  (Super)**; **AlpaSim** open-source — impact: "world model" now table stakes (H0/H6); Autobrains watch
  escalated; AlpaSim = usable sim asset (Tools&DevEnv) — https://www.electrive.com/2026/06/02/uber-and-autobrains-to-partner-on-munich-robotaxi-pilot-project/
- `[Opponent Analyzer]` [2026-07-24] [Opponent Analyzer] INFER (design-oracle, P8) — **Stop-Arm Gate** scenario (SC-04, W-03)
  shipped: H9 **violation rate rule_barrier 0.0 vs soft_prior 1.0** over the free-path temptation sweep;
  the barrier is invariant to temptation while the soft prior's line-crossing speed grows 3.0→9.6 m/s;
  OKRI toward the occluded child 80% lower at 4 B vs 15 B params (**11/11 offline tests**) — impact:
  **H9/H15**, first violation-rate contrast — see `2026-07-24-opponent-sweep-w3.md`
- `[Tools&DevEnv]` [2026-07-21] [root-cause] **The fleet monitor's blind spot is structural, not a bug**: every
  check in `.claude/skills/fleet-status/SKILL.md` grepped a **hardcoded** run/log name
  (`p0-sB01-realmix.log`, `arm_base.log`, `arm_kstep.log`, `pgrep -fc train_worldmode[l]`) —
  all belonging to runs that ended weeks ago. A grep that matches nothing prints nothing, and
  a monitor that prints nothing reports no anomaly. **Renaming a run silently blinds it**, and
  every arm since has been renamed → 4 recurrences, latest 2026-07-20 05:01 UTC (2 of 4 GPUs
  dead, the 04:55 probe clean). Fix = discovery, plus the rule **absence of evidence is an
  ALARM, not an all-clear** — impact: TOP-RISK/ops/all-agents —
  note `2026-07-21-fleet-probe-and-the-rerun-dual-sink-loss.md` §1
- `[Tools&DevEnv]` [2026-07-21] [built] **`tools/fleet_probe.py`** — discovers jobs from `ps` (grouped by
  `--out`, so a 6-proc fan-out = one run) and logs from the launcher's stdout redirect walked
  up the ppid chain; cross-checks GPU vs process table (`ORPHANED_GPU_MEMORY`,
  `GPU_IDLE_NO_TRAINER`), catches freezes two ways (`LOG_STALE` 15 min, `STEP_NOT_ADVANCING`
  via a state file), and measures disk with a real 100 MB `dd` (never `df`). Verdicts start
  UNKNOWN; a job with no discoverable log is **AMBER, never GREEN**. Measured live:
  **whole 4-pod fleet in 9.7-11.3 s**; 20 falsifiers 0.35 s — impact: TOP-RISK/ops/G-I — note §1
- `[Tools&DevEnv]` [2026-07-21] [measured] **pod2 (A40) idle with no trainer** on every probe run of 2026-07-21
  (0 %, 0 MiB, no job process; disk healthy 208-474 MB/s). Live instance of the class the old
  monitor missed 4x — impact: burn/M-1 resource mandate — note §1, escalated in STATE
- `[Tools&DevEnv]` [2026-07-21] [trap] **git-bash's MSYS `ssh.exe` deadlocks under `subprocess` pipes from a
  native-Windows Python** — the identical payload runs in **2.0-2.2 s from a shell** but hangs
  past 90 s from Python, reproducing **100 % on the two *training* hosts and 0 % on the two
  idle ones**, i.e. *it reads exactly like a fleet outage and is not one*.
  `C:\Windows\System32\OpenSSH\ssh.exe` ran the same payload on all 4 hosts in **0.7-2.5 s**.
  Prefer native OpenSSH on win32 for any Python-driven pod tooling — impact: all pod tooling —
  note §1 negative-results
- `[Tools&DevEnv]` [2026-07-21] [trap] **`subprocess.run(..., text=True)` corrupts every remote bash payload on
  Windows**: its stdin TextIOWrapper translates `\n` -> `os.linesep`, so every `fi` arrives as
  `fi\r` and bash dies with the misleading `syntax error: unexpected end of file`. A CRLF
  checkout does the same. Encode payloads to LF **bytes** — impact: all remote tooling — note §1
- `[Tools&DevEnv]` [2026-07-21] [trap] **`find /workspace -maxdepth 3` times out (>90 s) on the MooseFS pods**,
  and does so only on the *busy* ones — the naive form is blind precisely where it matters.
  Use per-dir `timeout 8 find ... -mmin -2880` — impact: pod tooling — note §1
- `[Tools&DevEnv]` [2026-07-21] [measured] **The `--rrd` + `--serve` dual sink is a 3,314x silent data loss.**
  rerun 0.34.1: `rr.save()` sets the file sink, `rr.serve_grpc()` **replaces** it (SDK's own
  docstring), so only the blueprint reaches the file. 200 windows x 3 arms x 256^2:
  rrd-only **10,593,179 B** (52,966 B/win, 299 win/s) vs dual-sink **3,196 B** (16 B/win).
  It survived because the file is **non-zero** — *non-zero is not non-empty; test emptiness
  only against a same-input single-sink baseline*. jpeg85 vs raw = **3.79x smaller for 17 %
  less throughput** (default is right). Guard shipped via intake
  `2026-07-21-rrd-dual-sink-guard/` — impact: P1 TanitResim/viz/G-T1 — note §2
- `[Tools&DevEnv]` [2026-07-21] [negative] **The documented rerun tee deadlocks**:
  `rr.set_sinks(FileSink, GrpcSink(url))` after `serve_grpc()` hangs indefinitely (killed at
  120 s, no output) — the GrpcSink connects back to the in-process server on the same thread.
  A real tee needs two `RecordingStream`s + explicit `recording=` per log call — impact: viz —
  note §2
- `[Tools&DevEnv]` [2026-07-21] [measured] **`rerun-sdk` is pinned in NO requirements file** anywhere in the repo
  (`stack/requirements*.txt`, `pyproject.toml` -> no match) although 0.34.1 is installed and the
  whole viz backbone depends on it. Also corrects a stale backlog premise: the "pin 0.34.1 +
  migrate, 1-2 h" work did not exist — 0.34.1 was already in the venv and `rr_log.py` (417 lines)
  already logs episodes — impact: reproducibility/G-T1 — note §2
- `[Tools&DevEnv]` [2026-07-21] [watch] **TerraZero still has no public code** (5-min check, backlog P1.0b).
  Project page `terra-applied.github.io`; **the GitHub org literally named `TerraZero` is an
  unrelated third party** — do not mistake it for Applied Intuition's release. Separately, an
  **AlpaSim E2E Closed-Loop Challenge 2026** exists (HF space) — a possible external yardstick
  if the docker-host blocker is ever cleared — impact: closed-loop fallback —
  https://terra-applied.github.io/
- `[Tools&DevEnv]` [2026-07-20] [measured] **The stack test suite has ZERO GPU coverage**: `grep -rl cuda
  stack/tests/` returns nothing across all 396/531 tests, while every trainer, eval and
  deploy tick runs on a GPU. Device/dtype placement, on-device batch-statistic leaks and
  CUDA-only NaNs were structurally invisible to CI. Closed by `tools/gpu_tripwire.py`
  (4 probes on the real model). Measured on the RTX 4060 (torch 2.11+cu128, fp32, 1.7 s):
  encode CPU-vs-CUDA **9.54e-07**, imagine **7.15e-07**, I2-on-device **1.66e-07**, 0
  non-finite grads; batch-1 encode **0.85–1.43 ms** (I8 proxy). Default tol 1e-3 = ~1000x
  headroom; a falsifier at tol=0 proves the probes can fail — impact: G-E/CI/I2/I8 —
  note `2026-07-20-ci-gate-v2-suite-manifest-gpu-tripwire-and-the-uncommitted-stack.md` §1–2
- `[Tools&DevEnv]` [2026-07-20] [root-cause] **40 uncommitted `stack/` paths on the shared Drive tree, 22
  UNTRACKED** — 12 test modules (~135 tests), 9 `tanitad/lake/*` + `eval/ckpt_compat.py`
  + `train/decorr.py`, 18 modified core files (`config.py`, `fourbrain.py`, `predictor.py`,
  `refa.py`, `flagship_losses.py`, 10 scripts). In no commit, on no branch. Found via a
  396-vs-531 collected discrepancy between the worktree and the Drive tree. **Strictly
  worse than D-026's unmerged branches** (those are at least pushed). `session_guard` v1
  called that tree clean because it only checked hub prefixes → source check added —
  impact: D-026/G-I/all-agents — note §4
- `[Tools&DevEnv]` [2026-07-20] [tooling] `git status --porcelain` **collapses a wholly-untracked directory
  to one `?? dir/` row** — fatal for any guard whose job is to name the missing files. Use
  `--untracked-files=all`. Caught by a falsifier before ship, not after — impact:
  tooling/session_guard — note §4
- `[Tools&DevEnv]` [2026-07-20] [built] **`ci_gate` v2** (`tools/ci_gate.py`, promoted out of stranded intake
  to repo-root tooling): adds a **SUITE_MANIFEST** (16 load-bearing modules pinned to a
  collected-count floor — a named-node tripwire only guards nodes somebody thought to name;
  whole modules vanish silently), `--min-total` (390), `--gpu-smoke off|warn|require`,
  `--json`. Skips stay green **unless a whole module is skipped**. Measured: both trees GATE
  PASS, **396/39.0 s** (off-Drive worktree) and **531/60.2 s** (Drive); 57 falsifiers 15.5 s
  — impact: G-E/CI — note §3
- `[Tools&DevEnv]` [2026-07-20] [measured] **Sharding NOT needed** (backlog condition was "<5 min or shard"):
  worst measured tree = **60.2 s, 5x under the ceiling**. Budgets set from measurement:
  per-test 15 s, wall 150 s. Caveat: timings are **contention-sensitive** — the same suite
  ran 65.0 s with a 14.90 s tall pole beside a second pytest process (vs 39.0 s / 8.02 s
  clean), so a concurrent agent run can false-positive the slow-test budget. This also
  re-scopes backlog P0.2: `test_replay`'s "10.86 s" was partly an I/O+contention artifact —
  impact: G-E/backlog — note §3.3–3.4
- `[Tools&DevEnv]` [2026-07-20] [verdict] **AlpaSim closed-loop on `tanitad-eval`: NO-GO** (executed
  2026-07-19 by the investigation agent — retires my P1.0). The eval pod is itself an
  unprivileged container with **no nested container runtime**, and AlpaSim's NuRec renderer
  ships only as `nvcr.io/nvidia/nre/nre-ga:26.04` — no source form. Policy/driver side GO
  (bare gRPC, adapter written); ~1.5 GB/scene, <2 GB VRAM would fit a proper host. Residual
  ask is infra (a docker-capable GPU host), not tooling — impact: P5/closed-loop/D-014 —
  `Benchmarks & Eval/Implementation/incoming/2026-07-19-alpasim-closedloop-v1/INTAKE.md`
- `[Tools&DevEnv]` [2026-07-20] [tooling] **Rerun 0.34.0/0.34.1 ships a Viewer MCP server** — an agent can
  see and interact with what the viewer renders, i.e. verify its own rollout overlay instead
  of asserting it. Also `VoxelGridMap`, transform-debug UI; **breaking API changes**
  (migration guide), pin **0.34.1** (live-stream stack-overflow fix). GO on a branch, est.
  1–2 h SDK bump + `corpus_overlay.py` migration + ~30 min MCP wiring — impact:
  WP-viz/TanitEval-viz-standard — [releases](https://github.com/rerun-io/rerun/releases)
- `[Tools&DevEnv]` [2026-07-20] [correction] Orin export should target **JetPack 7.2 (Jetson Linux 39.2,
  shipped 2026-06-02)**, not the 7.1 this KB recorded: 7.2 brings the **Orin family into the
  JetPack 7 line** (CUDA 13.2.1, TensorRT 10.16.2, unified Orin+Thor installer). NVFP4 is
  still Thor-only; Orin still targets FP8/INT8 — impact: C1/C2/P5 —
  [JetPack](https://developer.nvidia.com/embedded/jetpack)
- `[Tools&DevEnv]` [2026-07-20] [paper] **"Validate the Dream Before You Trust Its Verdict"** (arXiv
  2607.07196, RSS-2026 wksp): a world model used as a test ORACLE must be accredited first;
  L0–L4 admissibility ladder from VV&A/SOTIF. Key result: the model ranking higher on visual
  generation quality ranks **lower** on action-following — the citable external form of our
  open-loop-ADE ⊥ closed-loop finding (0.45 → 1.69 m). Seam: Benchmarks & Eval — impact:
  H15/eval — [abs](https://arxiv.org/abs/2607.07196)
- `[Tools&DevEnv]` [2026-07-20] [paper] **DynaDreamer** (arXiv 2607.13410): physics-informed ego-dynamics
  context that *modulates* a causal-Transformer WM, with a dynamics predictor keeping it
  synced **during rollout**; +28 % urban / +61 % highway, +73 % on an unseen chassis with no
  retraining. The principled generalization of our v0-as-3rd-action-channel fix (3.73 →
  0.83 m, speed-R² 0.965) and a direct lever on the longitudinal 83 %. No code, no stated
  scale → design input, not a dependency. Seam: Architecture — impact: H4/H25 —
  [abs](https://arxiv.org/abs/2607.13410)
- `[Tools&DevEnv]` [2026-07-20] [paper] **Orbis 2** (arXiv 2607.15898, Freiburg): hierarchical driving WM
  (coarse predictor + detail generator) trained **diffusion-forcing then teacher-forcing** —
  a reusable rollout-stability schedule that costs only a training-schedule change. Code +
  ckpts advertised, but generative-video scale → read the loop, do NOT run the weights —
  impact: H15/V3-hierarchy — [abs](https://arxiv.org/abs/2607.15898)
- `[Tools&DevEnv]` [2026-07-20] [paper] **TerraZero** (arXiv 2607.13028, Applied Intuition): procedural
  driving sim, **1.3 M agent-steps/s on one GPU**, pure-RL policies, tops InterPlan. Exactly
  our affordable closed-loop shape (no rendering) — but **no code released**, commercial
  vendor → WATCH, re-check in 2–4 weeks — impact: P5/closed-loop —
  [abs](https://arxiv.org/abs/2607.13028)
- `[Tools&DevEnv]` [2026-07-18] [built] `tools/session_guard.py` — the D-026 session-end stranded-work guard every
  agent runs (protocol-wired G-F/G-I). BLOCKS on uncommitted hub deliverables; WARNs on unmerged
  `agent/*` branches vs tip (`rev-list --count tip..branch`; current branch info-only) and on
  `incoming/*/INTAKE.md` with an unfilled `ORCHESTRATOR VERDICT` older than 3d. Tip defaults to HEAD
  (origin/main is diverged, `0f93b98`). Stdlib-only, 15 falsifiers 5.2 s. **Live-tree run flagged the
  real debt: 5 uncommitted hub files / 9 stranded branches / 5 stale INTAKEs (9d/5d)** — impact:
  G-F/G-I/D-026 — note `2026-07-18-session-guard-…-edgellm.md` §1
- `[Tools&DevEnv]` [2026-07-18] [bug-lesson] `git status --porcelain` parsing: a global `.strip()` on git stdout eats
  the leading status-column space of the **first** line (` M path` → `M path`) → fixed-offset `[3:]`
  path parse breaks silently for the first modified file. Use `.rstrip()` (preserve leading). Caught
  by the guard's own falsifier before ship — impact: all-tooling/G-T1 — note §1
- `[Tools&DevEnv]` [2026-07-18] [tooling] **AlpaSim (`NVlabs/alpasim`) + AlpaGym (`NVlabs/alpagym`) are now PUBLIC,
  Apache-2.0.** AlpaSim = microservice closed-loop AV sim (NuRec neural renderer, gRPC services,
  ready policies), eval data `PhysicalAI-AV-NuRec` HF. **AlpaGym RL default 10B policy needs 2 GPUs**
  → no-go single-4060/A40; reference-only until a 2×A40 pod or a <100M policy swap. AlpaSim is a
  single-A40 eval-harness smoke-test candidate (GPU-heavy NuRec → not a 4060 job) — impact:
  P5/H1/H11/closed-loop — [alpasim](https://github.com/NVlabs/alpasim) [alpagym](https://github.com/NVlabs/alpagym)
- `[Tools&DevEnv]` [2026-07-18] [deployment] **TensorRT Edge-LLM (JetPack 7.1)** = current ViT/VLM edge-export path
  (HF→quantize→ONNX→engine; `--visual_quantization fp8` for ViT). **NVFP4 (4× mem win) is Thor/SM110+
  (Blackwell) ONLY — Orin cannot run NVFP4, only FP8/INT4.** Rule: lock the target chip first (Orin→
  FP8/INT8; Thor→NVFP4). Alpamayo-R1-10B weights (~22GB) live on HF (open teacher); 34B-Super still
  unshipped — impact: C2/P5/Orin-path — [edge-llm](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)
- `[Data Engineering]` [2026-07-18] [measured/mix] **Curve-rebalance measured on real bytes (FLEET P0#3): the "74% straight" is a
  comma/HIGHWAY property, not a whole-corpus one.** 630 eps / 125,247 windows, D1 eval strata (`|net yaw@2s|`
  <5°/5-20°/>20°): **comma2k19 83.1% straight** (10.5/6.4 gentle/sharp), **PhysicalAI 56.0%** (23.4/20.6) —
  urban is already at the 55-60% target. Natural window pool = **63.9% straight**; the fleet's ~74% ≈ a comma
  0.65-0.70 mix (comma 0.70/pai 0.30 → 75.0%). Two quantified levers to 57.5%: **source-mix** (+10pp comma ≈
  +2.7pp straight → shift to urban/ZOD/PandaSet = the ZOD rationale, in numbers) and **window turn-weighted
  sampling** β=s(1-t)/(t(1-s)) = **1.31** (natural pool) → 2.22 (comma-0.70). Recipe verified by construction;
  a drift-guard test pins the strata to `driving_diagnostic.py`. Training-recipe change → D-018 ESCALATE before
  flipping the trainer — impact: top-risk/D1-straight/H25/G3 — `2026-07-18-curve-rebalance-measured-and-idm-lit.md` §2
- `[Data Engineering]` [2026-07-18] [arXiv/lit] **IDM/latent-action + intrinsic-canon delta (seeds G2/H7, no status change P8):**
  Sensorimotor World Models (2606.20104, IDM-regularizer stops collapse + preserves the action — the pose-less
  pseudo-label recipe); ACID (2607.02403, action-cycle-consistency = a per-clip pseudo-label QUALITY gate for the
  H7 bridge); Latent-WAM (2603.24581) / DriveWAM (2605.28544) driving latent world-action models; "What Do Latent
  Action Models Actually Learn?" (2506.15691, LAM failure-mode caution — read before trusting any latent
  pseudo-label); **X-Lens (2607.12993)** intrinsic-guided canonicalization to a reference camera + calibration
  tokens = the closest recent work to our D-016 `f_eff=266` / H17 unified-FOV. New datasets → Benchmarks seam:
  global urban dashcam corpus (2604.01044, pose-less, curve/IDM candidate, license TBD), DrivingGen (2601.01528),
  ScenePilot-4K (2601.19582), LiAuto-DriveAction (HF) — impact: H7/G2/D-016/H17/D-028 — same note §4
- `[Data Engineering]` [2026-07-18] [measured/loader] **ZOD loader SHIPPED + geometry falsifier PASS — the #1 owned real-urban ingest.**
  ZOD front is a **Kannala-Brandt fisheye** (8 MP, 3848×2168, **HFOV 120°**, 10 Hz) and KB's radius
  `r(θ)=f(θ+k1θ³+k2θ⁵+k3θ⁷+k4θ⁹)` IS exactly `calib.FThetaIntrinsics.poly` (odd-power) → `kb_to_ftheta`
  reuses the proven f-theta crop path with **zero new geometry math** (confirms OWN_DATASET_PLAN's
  "fisheye→ftheta_*" with numbers). **Pre-registered falsifier ANSWERED (grounded on the published 120°
  spec, robust to real KB coeffs): f_eff=266.0 px, observed_frac=1.00, drop_in=True** — a 120° fisheye crops
  INWARD to canonical ~51.4° so nothing is padded. ZOD is **geometrically UNBLOCKED** (contrast PandaSet,
  height-bound f_eff 467; **ZOD needs NO calib.py R1 change**). Narrow-40° witness falsifies (observed_frac
  0.34) so the ≥0.5 gate is not vacuous. Real CAN steer + OxTS RT3000 (0.01 m/0.1°, @100 Hz) → OxTS heading
  drives yaw offset-free (better than PandaSet's motion-heading fallback); `zod_signals` reuses tested
  `cosmos_drive.poses_to_signals`. CC-BY-SA → SEPARATE shard (ShareAlike firewall). Intake `2026-07-18-zod-loader/`
  (19✓ standalone) + runnable real-bytes job card (blocked only on ZOD ACCESS, escalated) — impact:
  OWN_DATASET_PLAN §7#1 / FLEET_REVIEW P0#1 corpus-diversity / H4 arm-B / H7 — `2026-07-18-zod-loader-and-geometry-falsifier.md`
- `[Data Engineering]` [2026-07-18] [lit/seam] **New driving-WM benchmarks + a new urban dashcam corpus.** WorldLens (CVPR-2026
  Oral, WorldLens-26K human-rated realism/plausibility/safety) + DrivingGen (arXiv 2601.01528, generative-WM
  bench: trajectory plausibility + controllability ≈ our D2/D4) → **Benchmarks&Eval seam** (D-028). NEW
  **"A global dataset of continuous urban dashcam driving"** (arXiv 2604.01044) — a candidate curve-rebalance
  urban source (BACKLOG P0#3); **license + actions-availability probe queued** (owned-tier add vs YouTube-class
  barrier unknown until probed) — impact: D-012/curve-rebalance/H4 — same note §1
- `[Architecture & Inference]` [2026-07-18] [repo/measured] **OPERATIVE flagship-speed @19k: the σ-dissipation + attractor collapse
  REPRODUCES (drops the pre-reset caveat); readout isotropy is CONVERGING toward admissibility.** Re-ran
  E1+E2 (backlog P0.1) on `flagship-speed` (WorldModel flagship4b, action_dim=3, step 19000) on the eval
  pod A40, canonical PhysicalAI val, 2 seeds, $0. **E1 (blind K-step rollout, 320 windows):** the P0.1
  falsifier "speed+jerk fixed it" is **NOT met** — cos_rollout dies to chance by **k3** (0.232→0.016→neg),
  σ_hidden nets **−9.461→−9.564** (more confident as it decays; *lower* absolute σ than the −7.8 pre-reset
  ckpt = worse temporal calibration), attractor inter-sample cos **0.219→0.805** (sharper than 0.57
  pre-reset). **freeze-1 holds 0.232→0.213 FLAT across 8 horizons, 7× persistence** → parallel-horizon is
  the safe operative mode, confirmed on the shipping model. **Refinement:** σ is *spatially* calibrated
  (calib_gap +0.37 hidden>visible; per-cell err↔var corr +0.29–0.43) but *temporally* anti-calibrated —
  the design target narrows to a **horizon-aware** σ, not a spatial rebuild. **E2 (orthogonality, 7,964
  latents):** `iso_ratio_active` **0.254→0.546** (crossed 0.5), `cond_active` **218→61**, `rms_offdiag`
  0.42→0.32 — SIGReg converging exactly as the 07-17 note predicted; still **NOT-YET-ADMISSIBLE** (offdiag
  0.32>0.1 → LeJEPA optimal-planning corollary still withheld). active_k≈19, cov_eff_rank≈30 ≪ 2048 →
  **readout capacity is NOT the D1 bottleneck (G1), reaffirmed on the operative model.** No config change
  (D-018); decision-grade re-run at @30k is turnkey (both scripts, ~2 min pod). — impact: H15 / H11 / D8 /
  H3 / D-021 / G1 — `../Research/2026-07-18-operative-flagship-blind-rollout-and-orthogonality.md`
  + `../Implementation/belief_rollout_diagnostic/blind_rollout_flagship.py`
- `[Architecture & Inference]` [2026-07-18] [ICLR2026 openreview pZuZWRuPyi] **HAUWM — "Learning to Be Uncertain: Pre-training World
  Models with Horizon-Calibrated Uncertainty" is the direct design anchor for backlog 0b-A** (the fix for
  E1's exact failure). A probabilistic-ensemble WM predicts frames at **randomly sampled future horizons**
  and a **Horizon-Calibrated Uncertainty (HCU) loss** shapes the latent so **predictive variance GROWS with
  horizon** — precisely the property my 07-18 E1 found MISSING on the flagship (σ dissipates instead). Two
  routes for 0b-A: (a) HCU-style horizon-sampled variance target on our single logvar head (cheaper, no
  ensemble; our ImaginationField already emits per-cell logvar — just supervise it against realized
  multi-horizon error so it must rise); (b) small ensemble à la **ELVIS** (2605.04709, ensemble-calibrated
  latent imagination for long-horizon visual MPC) if the single head can't express growing σ. **CAVEAT
  (P8): full text is behind OpenReview verification — mechanism is from the search abstract, the exact HCU
  loss form is UNVERIFIED; fetch the arXiv mirror before porting.** — impact: H15 / H11 / D8 / D9 (backlog
  0b-A) — https://openreview.net/forum?id=pZuZWRuPyi · https://arxiv.org/pdf/2605.04709
- `[Tools&DevEnv]` [2026-07-17] [built] `ci_gate` — one-command self-testing pytest gate (backlog P0.1). Fails on
  pytest failure OR **collection error** (defers to pytest exit code → never a false GREEN), per-test
  >15 s, wall >90 s, or a missing/failing required tripwire (default the I2 encoder batch-consistency
  test). Stdlib-only, OS-agnostic (`ci.ps1` Win wrapper / `python ci_gate.py` on pod). Measured:
  11/11 falsifiers; catches the live broken suite in 3.9 s; clean suite 343+2skip 47–57 s — impact:
  G-E/CI/D-004 — intake `2026-07-17-ci-gate/`, note `2026-07-17-…-bench2drive-speed.md` §2
- `[Tools&DevEnv]` [2026-07-17] [root-cause] The stack suite was **RED for every agent** on 2026-07-17: an untracked
  TDD test `tests/test_physicalai_rig.py` (Data-Eng D-016 R1 two-rig fix) imports `ftheta_horizon_row`
  / `ftheta_project_ray` / `ftheta_crop_box` + `center=`/`per_clip=` that committed `calib.py` never
  shipped → `pytest` exit 2 at collection, 0 of 343 tests run. Fix-forward = `ci_gate` makes it a hard
  gate; remediation (land calib impl or xfail) is Data-Eng/orchestrator — impact: G-E-all-agents —
  note §1
- `[Tools&DevEnv]` [2026-07-17] [tooling] Agent tooling on the Windows dev box must be **ASCII-clean stdout**: `ci_gate`
  v1 crashed with `UnicodeEncodeError` printing `✓/✗` under the cp1252 console. Use ASCII markers (or
  force UTF-8) in any script an agent/CI runs on this box — impact: G-T1/all-tooling — note §2
- `[Tools&DevEnv]` [2026-07-17] [opponent] NVIDIA **Alpamayo 2 Super = 34 B** (corrects prior KB "32 B"), closed-loop via
  AlpaGym on AlpaSim; GitHub/HF "this summer". NVIDIA shipped a "post-train AV in closed-loop with
  Alpamayo" dev blog + an Alpamayo-1 trajectory-latency paper (arXiv 2605.08975, on our C2/P5 edge
  thesis). Verdict unchanged: Phase-1 cloud (40–60 GB) — impact: P5/H1/opponent —
  [newsroom](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis)
- `[Tools&DevEnv]` [2026-07-17] [benchmark] **Bench2Drive-Speed (Mar 2026)** grades closed-loop *speed customization* —
  directly validates the program's speed/scale reset (v0-as-action-channel, probe R² 0.61→0.965) and is
  a Phase-1 closed-loop eval target. **Dev10** = 10-clip quick-dev subset (fast iteration); Bench2Drive-VL
  (Oct 2025) = closed-loop VLM QA. Seam: Benchmarks&Eval owns — impact: H4/speed/eval —
  [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- `[Opponent Analyzer]` [2026-07-17] [Waymo/NHTSA] FACT — Waymo recalled **3,871 robotaxis (2026-06-18)** for driving into
  freeway **construction zones** (unrecognized ramp-closure signs; drove between lane-closure cones);
  freeway autonomy suspended, 20+-city expansion frozen — impact: **W-01 → H15/H9/H1**; drives the new
  Work-Zone Phantom scenario — https://www.cnbc.com/2026/06/18/waymo-nhtsa-voluntary-recall-robotaxis-entered-freeway-construction-zones.html
- `[Opponent Analyzer]` [2026-07-17] [NTSB] FACT — HWY26FH008 (2026-01-23, Santa Monica): Waymo I-Pace struck a 9-yo
  pedestrian in a school zone; ADS **detected + braked heavily but late**; NTSB examining sudden-VRU
  anticipation. Distinct from the separate school-bus stop-arm probe (CLAIM: one case = human error)
  — impact: **W-02/W-03 → H15/H9 (LOPS/OKRI/LAL, D9)** — https://www.ntsb.gov/investigations/Pages/HWY26FH008.aspx
- `[Opponent Analyzer]` [2026-07-17] [NHTSA SGO] FACT — Jun'25–May'26 window: 825 ADS incidents; **Waymo 697** (1 fatality,
  23 hospitalizations, 51 minor, 613 property-only) — impact: weakness-evidence corpus, H0/H6 — https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting
- `[Opponent Analyzer]` [2026-07-17] [CA DMV] FACT — Dec'24–Nov'25: Waymo 19,234 mi/disengagement (3.35 M mi), Zoox 60,682;
  DMV **proposes to replace the disengagement metric in 2026** — impact: **W-07 narrative**, aligns with
  Benchmarks&Eval open-loop⊥closed-loop — https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/disengagement-reports/
- `[Opponent Analyzer]` [2026-07-17] [Tesla/NHTSA] FACT — NHTSA engineering analysis (Mar 2026, pre-recall): camera-only FSD
  **fails under degraded visibility (glare/airborne obscurants)**; robotaxi in Miami (2026-07-03), TX
  fleet ~42 vs Waymo 577 — impact: **W-04 → H11/H15/H2** — https://www.automotiveworld.com/news/tesla-robotaxi-fleet-hits-25-as-musk-defers-scale-to-fsd-v15/
- `[Opponent Analyzer]` [2026-07-17] [Wayve] FACT — Series D **$1.2 B (Feb'26), $8.6 B** post-money ($1.5 B total); **GAIA-3 =
  15 B latent-diffusion WM for offline eval**; on-car AV2.0 = monolithic E2E cam+radar, mapless
  (corrects kickoff "$2.8 B") — impact: **W-05 → H1/H3/H5 (CNCE)** — https://wayve.ai/press/series-d/ , https://wayve.ai/thinking/gaia-3/
- `[Opponent Analyzer]` [2026-07-17] [NVIDIA] FACT — **Alpamayo 2 Super = 32 B reasoning VLA** on Cosmos (Chain-of-Causation);
  open dataset **1,700+ h / 25 countries / 2,500+ cities** — impact: **frenemy/supply chain; W-05 foil**;
  32 B on-car = anti-efficiency — https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development
- `[Opponent Analyzer]` [2026-07-17] [Pony.ai] FACT — Q1'26 total rev $34.3 M (+145% YoY), robotaxi rev $8.6 M (+395% YoY) vs
  3,500-vehicle target; Croatia (first EU commercial robotaxi) + Dubai — impact: **W-06 → H3/H7** (thin
  unit economics) — https://mlq.ai/news/v2/pony-ai-q1-revenue-more-than-doubles-to-343m-as-robotaxi-sales-surge-nearly-fivefold/
- `[Opponent Analyzer]` [2026-07-17] [Momenta] FACT — HK IPO ~$752 M, ~$9 B valuation (trading 2026-07-08), GM+Tencent;
  two-leg L2++/robotaxi; Abu Dhabi + Munich 2026, Uber L4 pilot — impact: strategy-split weakness locked
  by public markets — https://technode.com/2026/06/30/momenta-launches-hong-kong-ipo-with-gic-fidelity-and-blackrock-as-cornerstone-investors/
- `[Opponent Analyzer]` [2026-07-17] [Autobrains] FACT — $140 M+ funding (BMW/Toyota/Continental/Temasek); "Liquid AI" +
  agentic, edge-cases-with-less-compute on standard sensors; L2+/ADAS only — impact: **narrative-overlap
  watch → own L4/WM ground + pre-empt CNCE** — https://autobrains.ai/about-us/
- `[Opponent Analyzer]` [2026-07-17] [arXiv] FACT/INFER — 2026 **latent-WM/JEPA-for-driving surge** (survey 2603.09086;
  Drive-JEPA 2601.22032; Metis "efficient world-action model" 2606.15869; GraphWorld 2606.16274; IDOL
  2605.31476; +more) — impact: H3 externally validated but **"latent WM" no longer differentiating** →
  moat = hierarchy+efficiency+imagination+self-monitoring; deep-read Metis next run — https://arxiv.org/abs/2603.09086
- `[Data Engineering]` [2026-07-17] [measured/tool] **D-016 R1 pinhole rectify UNBLOCKS the owned real-urban tier.** New primitive
  `pinhole_rectify` (grid_sample rectify-to-canvas, Brown-Conrady undistort + pad; mirrors the existing fisheye
  `ftheta_undistort`) lands `f_eff=266` **exactly by construction** where the square-crop is height-bound.
  Measured (grounded real intrinsics): **PandaSet front 467→266.0** (drop-in), at a cost of **37.7% masked
  periphery** (native VFOV 30.7° < canonical 51.4°; sky/hood band unobserved, road band retained) + **109px k1
  barrel distortion corrected**; comma2k19 reference untouched (266.0, 99.6% observed). **New ingest rule:** gate
  every source on `observed_frac ≥ ~0.5` — Udacity-like falsifies at 0.13 (narrow FOV = 87% mask). Undistort
  correctness: fwd↔iterative-inverse <1e-4, checkerboard recovery corr>0.9. Contract-drop-in (G-D2). Intake pkg
  `2026-07-17-d016-r1-pinhole-rectify/` (9✓). Coverage map: pinhole (PandaSet/Udacity/comma) → this; fisheye
  (ZOD KB/PhysicalAI/Cosmos f-theta) → existing `ftheta_*` — impact: D-016/G1/OWN_DATASET_PLAN/H17 —
  `2026-07-17-d016-r1-pinhole-rectify-unblocks-owned-real-urban.md`
- `[Data Engineering]` [2026-07-17] [measured/pitfall] **A8 on 12 real comma-val eps (3,600 frames): 0.0596@0.05 / 0.0240@0.10**
  (curr-frame slice) — reproduces the 2026-07-07 baseline (~0.053/0.012), low-consequence highway regime holds
  on held-out val. **Harness pitfall:** `stats.frame_change_fraction` assumes float [0,1] but the epcache stores
  uint8 [0,255] → a direct caller gets a meaningless ~0.74 (uint8 subtract wraps); convert via `to_float_frames`
  first. BACKLOG: make `stats` uint8-safe — impact: H3/A8, stats harness — same note §4
- `[Benchmarks & Eval]` [2026-07-17] [this run / measured] **The ego-status shortcut ceiling on OUR data = avg L2 0.66 m
  (comma-hwy, metric-BEV, held-out by clip).** A no-vision ~20-param ridge from ego-status history scores
  0.144/0.552/1.256 m @1/2/3s — statistically tied with CTRV (0.656) — the AD-MLP shortcut (2312.03031)
  reproduced on comma. **`skill_score = model_L2 ÷ 0.66 m` now defined in leaderboard-comparable units.**
  cosmos-urban: the *learned* shortcut (1.19 m) beats the fixed kinematic floor (1.34) — impact: G1 /
  validation strategy / D1 — `../Implementation/incoming/2026-07-17-openloop-l2-egostatus-shortcut/`
- `[Benchmarks & Eval]` [2026-07-17] [arXiv 2312.03031, CVPR'24 / this run] **comma highway is 73.9 % straight — identical to
  nuScenes' 73.9 %** (the ego-status-critique figure). Our open-loop val inherits the *exact* shortcut
  pathology: aggregate open-loop L2 is dominated by trivial straight cruising → a **weak capability test**
  (community-unit restatement of "10–15× worse than CV" + 2605.00066). Verdict must be per-stratum
  `skill_score` + closed-loop, never an aggregate open-loop L2 — impact: G1 / DIAGNOSTIC §A/C — https://arxiv.org/abs/2312.03031
- `[Benchmarks & Eval]` [2026-07-17] [protocol] **nuScenes L2 has two undisclosed averaging conventions** — `pointwise` (UniAD:
  L2 at exactly t) vs `cumulative` (ST-P3/VAD: mean up to t); they differ ~2×. Any TanitAD L2 row (and any
  competitor row we cite) must state which — impact: G-B1 leaderboard hygiene — `openloop_l2.py`
- `[Architecture & Inference]` [2026-07-17] [repo/measured] **Blind K-step belief rollout DISSIPATES uncertainty + collapses to an
  attractor — the H11/D8 σ-trigger is anti-calibrated past 1 step.** Rolled the trained 1-step
  ImaginationField fully blind on real comma2k19 (step-6500 base250cam ckpt, 4060, 2 seeds). Hidden-cell
  centered-cosine fidelity **0.357 (k1) → 0.011 (k4, = chance) → negative**; it **beats the persistence
  baseline only at k=1** and falls below it from k≥2. Epistemic σ (hidden log-variance) **falls
  monotonically −7.79 → −8.55** (more confident as predictions become worthless = FALSE confidence);
  belief energy **collapses ~11× by k4** (0.101→0.008) while inter-sample cosine **rises 0.21→0.57** →
  every sample drifts to a **common attractor** (true-token energy flat ~0.33, so it is the model, not
  the scene). **The cause is the recursion, not the 1-step prediction:** *freezing* the k=1 imagination
  and holding it retains **~0.25 cosine FLAT across all 8 horizons** and beats persistence throughout.
  Confirms the 2026-07-15 UWM-JEPA risk and the **"Biased Dreams" (2604.25416) attractor prediction**,
  measured. Two D-018 responses (escalate, don't execute): (A) train multi-step belief rollout (0b build);
  (B) operate imagination **parallel-horizon (non-autoregressive)** from the last real obs — freeze-1
  shows (B) recovers fidelity for free (recommended default). **Cap the operative H15 self-monitor at
  1-step until a multi-step σ is validated.** No H15 status change (P8, pre-reset directional ckpt) —
  impact: H15 / H11 / D8 / D9 — `../Research/2026-07-17-blind-rollout-uncertainty-dissipation-and-readout-orthogonality.md`
  + `../Implementation/belief_rollout_diagnostic/`
- `[Architecture & Inference]` [2026-07-17] [repo/measured] **Readout orthogonality — VERIFIED the stranded 2026-07-10 instrument
  (not a duplicate); D-021 = subspace ID, NOT "optimal planning" on the pre-reset ckpt.** While drafting a
  3b instrument I found a theoretically-superior one already built 2026-07-10 but **never merged** (branch
  `worktree-agent-arch-inf-20260710`); **withdrew my draft**, ran the prior `orthogonality_report`
  unchanged on the step-6500 ckpt (n=2600 real states > S=2048) → **reproduces exactly:** active_k=23,
  **iso_ratio_active 0.254** (< 0.5), cond_active 218, rms_offdiag 0.424, cov_eff_rank 26, verdict
  **NOT-YET-ADMISSIBLE**. Key correction: **global** isotropy ~0 is over-provisioning **by design**, NOT a
  failure — the theorem-relevant read is the **active-subspace** isotropy (my draft lacked this). My
  independent global read (isotropy 0.000, off-diagonal 0.999) corroborates over-provisioning from the
  coordinate angle. Two instruments now agree the readout is over-provisioned (op-rank ≈43, repr active-rank
  ≈23–26 ≪ 2048) AND not yet orthogonal → latent *capacity* is not a D1 bottleneck. SIGReg slice-Gaussianity
  ≠ active-subspace isotropy (cond 218) → whitening lever (D-018 escalate). **Process: flagged the stranded
  07-10 instrument for orchestrator merge (3rd week unmerged).** — impact: H3 / D-021 / D-008 —
  `../Implementation/orthogonality_verification/`
- `[Benchmarks & Eval]` [2026-07-16] [arXiv 2605.00066] Cross-benchmark study (15 methods): ADE/FDE have **no reliable
  correlation** with closed-loop Driving Score; NAVSIM PDMS correlates positively but **non-monotonically**
  with Bench2Drive DS (ranking inversions); fully-paired subset only n=8 — impact: validation strategy /
  D1–D6 (closed-loop arbitrates) / justifies custom suite — https://arxiv.org/abs/2605.00066
- `[Benchmarks & Eval]` [2026-07-16] [arXiv 2506.04218] NAVSIM v2 pseudo-sim (3DGS-augmented) R²≈0.8 vs closed-loop (0.7 pure
  open-loop); PDM-Closed **EPDMS=51.3** navhard (Mar-2026 snapshot); criticized as non-reactive, short-
  horizon, PDMS over-weights progress/comfort/TTC (thin on safety-critical occlusion = our OKRI/LOPS niche)
  — impact: LEADERBOARD context / metric-gap — https://arxiv.org/abs/2506.04218
- `[Benchmarks & Eval]` [2026-07-16] [Bench2Drive] Closed-loop CARLA: 220 short routes, **one safety-critical scenario each**, 44
  categories×23 weathers×12 towns; SOTA ctx TF++/VLAAD-MIL DS 86.97/SR 71.97, ADT 77.90/55.0 — impact:
  closed-loop competitor rows / weak-spot scenario template — https://github.com/Thinklab-SJTU/Bench2Drive
- `[Benchmarks & Eval]` [2026-07-16] [multi-source] CARLA closed-loop **~5 DS run-to-run seed variance** for the same model →
  gate claims need mean±CI over ≥3 seeds, CIs must separate to claim "beats baseline" — impact: G-B /
  validation rigor
- `[Benchmarks & Eval]` [2026-07-16] [UNECE WP.29, June-2026] Global ADS GTR adopted: SMS + credible-testing/safety-case (incl.
  validated virtual toolchains) + ISMR + **DSSAD** (standard format / retrievable via electronic interface
  / tamper-evident) — impact: REGULATION_TRACE / H10–H12 — Ressources/ECE-TRANS-WP.29-2026-139e.pdf
- `[Benchmarks & Eval]` [2026-07-16] [Deep Think 14 / this run] Custom metric suite implemented (LAL/TMS/OKRI/CNCE/LOPS +
  trajectory seam), 22 tests on analytic ground truth; plugs into the D1–D3 gate runner's `extra_metrics`
  seam (verified live) — impact: WP6 / G0.6 — `../Implementation/incoming/2026-07-16-eval-metric-suite/`
- `[Data Engineering]` [2026-07-15] [measured/gate] **WorldModel-Synthetic-Scenarios is POSE-LESS** (BACKLOG P0.1 gate CLOSED on
  real bytes): each clip = `<family>/<clip>/{description,video}`; `video/` = **7 camera mp4s** (front_wide,
  front_tele, 3 fisheyes, rear_left/right) @24 fps ~462 frames; `description/<cam>.json` = a **Qwen2.5-7B
  caption + `{weather,time_of_day,surface_type,region}`**, NOT a pose. No vehicle_pose/CAN/trajectory anywhere.
  → the "near-zero cosmos-mirror" assumption is DEAD; loader path is (a) video-only, (b) **IDM/H7 pseudo-label**
  (Phase-1, needs a trained inv-dyn head), or (c) **usable-today semantic-label index** (captions+metadata →
  BACKLOG P1 2d + SC-02/05/06). Families emergency/lanechange/nudging/pedestrian/weather_degradation. Do NOT
  fetch the 8.3 TB pixels until (b)/(c) scheduled — impact: H7/D-014/D-022/BACKLOG — `2026-07-15-worldmodel-pose-gate-and-pandaset-geometry.md` §1
- `[Data Engineering]` [2026-07-15] [measured/loader] **PandaSet loader shipped (intake, 16✓) + a grounded D-016 GEOMETRY BLOCKER.**
  CC-BY-4.0 real-urban adapter (plan §7 #2), reuses cosmos geometry (motion-heading 4×4 → poses_to_signals),
  I7≡comma2k19, I3 seq-split. Grounded on REAL front calib (arXiv 2112.12610: fx=1970.01, 1920×1080, k1=−0.589):
  centered square-crop canonicalization is **height-bound** (ideal crop 1896 px > 1080 frame height) → lands
  **f_eff=467 px vs canonical 266** (~1.75× scale mismatch) → NOT drop-in; **rule: any fx>1122 px on a 1080-tall
  frame is height-bound.** Distortion k1=−0.589 also ignored by the pinhole path. Loader **fails loud**
  (GeometryError) so it can't pollute the mix. Fix = D-016 R1 pad-crop+undistort in calib.py — a **prerequisite
  for the whole owned real-urban tier** (ZOD/Udacity hit it too), promoted from "R1 nicety" to blocking — impact:
  H7/H4/D-016/OWN_DATASET_PLAN — same note §2
- `[Data Engineering]` [2026-07-15] [HF sweep/D-012] New `nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic` (card-only, no data
  payload yet — watch; sibling of Cosmos-DD). `Newsflare/…-autonomous-driving-videos` = commercial stock video
  (copyright barrier, excluded, like OpenDV). No new ungated real-AV video corpus → owned real-urban gap stays
  **ZOD-shaped** (plan §7 #1). Literature: IDM/latent-action-in-the-wild now dense (2601.05230, 2602.16229,
  LatentVLA, FLAM) → frozen-encoder IDM+WM on unlabelled video is the standard recipe → makes pose-less corpora
  (WorldModel-Synth, YouTube) trainable via the comma/ZOD real-CAN bridge — impact: H7/D-012 — same note §3–4
- `[Benchmarks & Eval]` [2026-07-15] [this run / measured] **The honest trivial-baseline floor is CTRV best-of-3 ≈ 0.056–0.06 m@1s,
  not the single-CV 0.28 m** the driving diagnostic used. 26 132 anchors (comma-val + Cosmos-DD), 10 Hz. CV
  is the weakest kinematic null on curves (gentle CV 0.275 vs CTRV 0.060 = 4.6×); CTRV wins 55–58 % of anchors.
  → model held-out 6.44 m = **~115× floor** (not 10–15×), verdict direction reinforced; **D1 gate should use
  `skill_score` = model_ADE ÷ per-stratum best-of-3 floor** — impact: D1 gate / DRIVING_DIAGNOSTIC §A / WP6 —
  `../Implementation/incoming/2026-07-15-baseline-floor/`
- `[Benchmarks & Eval]` [2026-07-15] [this run / measured] **Curvature stratification must be speed-gated** (v ≥ 2 m/s): 12.4 % of
  comma anchors are near-standstill (median 0.01 m/s) where κ=yaw_rate/v is singular → GNSS yaw-jitter
  mislabels them "sharp" with a spurious 0.003 m floor. The framework's `driving_diagnostic` §C strata are
  standstill-polluted without the gate — impact: diagnostic protocol / P8 — same intake.
- `[Benchmarks & Eval]` [2026-07-15] [this run / measured] **The ungated Cosmos-Drive-Dreams sample is a poor maneuver source:**
  95.8 % straight, 1.8 % gentle, **0 % genuine sharp**, median 12.9 m/s. comma-highway carries MORE real curve
  content (12 % gentle + 0.8 % highway-speed sharp). Refines the 2026-07-13 note (Cosmos-DD = scene-diversity
  only) + framework §D2 (curve-scarcity remedy needs semantic-label survey, not more Cosmos-DD) — impact:
  Data-Eng curve-scarcity / backlog #3 — same intake.
- `[Benchmarks & Eval]` [2026-07-15] [arXiv 2506.04218 / NAVSIM v2] **NAVSIM v2 uses a constant-velocity agent as a triviality
  FILTER** (removes frames a CV agent solves with PDMS>0.8). Community precedent that the CV floor is
  load-bearing AND stratum-sensitive → validates our per-stratum best-of-3 skill denominator — impact:
  validation strategy / D1 gate — https://arxiv.org/html/2506.04218v1
- `[Benchmarks & Eval]` [2026-07-15] [arXiv 2510.18552 / 2605.18059] **New occlusion-robustness benchmarks (D-028 seam, ours):**
  **Occluded-nuScenes** (multi-sensor: 4 camera + parameterised radar/LiDAR occlusion types) — public,
  citable stressor for our OKRI/LOPS suite; **Bench2Drive-Robust** (closed-loop AD under occlusion **and
  inference latency**; SimLingo degrades sharply) = our exact edge pair OKRI/LOPS × CNCE — impact:
  LEADERBOARD watch / occlusion suite — https://arxiv.org/abs/2510.18552 · https://arxiv.org/html/2605.18059
- `[Benchmarks & Eval]` [2026-07-15] [arXiv 2605.31476] **IDOL — Inverse-Dynamics-Guided Future Prediction** — external support for
  the diagnostic's #1 root-cause lever (inverse-dynamics / ego-motion supervision grounds the latent).
  Pointer to Architecture; no status change (P8) — impact: DRIVING_DIAGNOSTIC root cause / H1 — https://arxiv.org/pdf/2605.31476
- `[Architecture & Inference]` [2026-07-15] [repo/measured] **The flagship H15 imagination edge is NOT dark — the log is unfaithful**
  (resolves the 2026-07-14 program-report §8 WATCH `h15=0.0`). GPU diagnostic on the exact code path
  (`train_flagship4b.h15_loss` + `flagship4b_smoke_config`): imagination module **built** (22.06 M params,
  `h15.enabled=True`), gradient **reaches** it (L1 44.6) **and the encoder** (L1 36.7), fire rate **0.4525**
  ≈ `mask_prob` 0.5, mean loss when fired **0.611**. `h15=0.0` is a **logging artifact** — `log["h15"]`
  records the LAST accum micro, 0.0 whenever its gate didn't fire; **46.3 % of all log rows falsely read
  `h15=0.0` while the edge trained**, true idle only 6.3 % (theory (0.5)⁴=0.0625 ✓). **Do NOT change the
  trained config** (would chase a phantom, D-018 restraint). Fix = observability: an accumulation-window
  meter (`h15`/`h15_fired`/`h15_fire_frac`) shipped as intake (6✓); `h15_fire_frac→0` is now the *real*
  dark-edge alarm — impact: H15 / D9 / D8 — `../Research/2026-07-15-h15-imagination-edge-not-dark-and-belief-space-rollout.md`
  + `../Implementation/incoming/2026-07-15-h15-logging-fidelity/`
- `[Architecture & Inference]` [2026-07-15] [repo/measured] **H15 imagination edge is affordable per tick** (CNCE/Efficiency moat).
  Batch-1, RTX 4060, flagship4b scale (263.44 M total, imagination 22.06 M = 8.4 % of params); latency
  weight-value-invariant so untrained instantiation valid for timing. **fp32:** encode 7.67 ms /
  imagination 2.25 ms / predictor 5.52 ms → core tick 13.18 ms → **H15 = 17.0 % of core**. **fp16:** 4.26
  / 1.35 / 6.40 → 10.66 ms → **12.7 %**. So the A9 self-monitor adds ~1.3–2.2 ms/tick (~roughly its param
  share), only when engaged → no efficiency-moat regression. Honest: **fp16 makes the small predictor
  SLOWER** (6.40 vs 5.52; batch-1 launch/convert-bound, not tensor-core-bound) — the fp16 win is entirely
  in the ViT tower (matches Prod-Opt "TRT-fp16 the tower"); eager un-fused, so absolute ms is an upper
  bound, the fraction is robust — impact: H5 / CNCE / H15 —
  `../Implementation/h15_logging_diagnostic/results/2026-07-15-h15_latency.json`
- `[Architecture & Inference]` [2026-07-15] [arXiv 2605.25313] **UWM-JEPA — belief-space imagination WM.** Density-matrix latent +
  learned unitary predictor imagine multiple compatible hidden futures; *"the construction preserves the
  joint-state spectrum exactly during rollout, so the predictor itself cannot dissipate the represented
  uncertainty."* Numbers: hidden-velocity 5-step forward-sim **0.77 vs 0.53** (LSTM-JEPA); blind rollout
  loses **<10** probe-R² pts short-horizon vs **41/68** baselines. Translations for our H15 (sector-mask
  1-step + advection + per-cell σ): (a) **we train imagination at 1 step only** → multi-step belief rollout
  (where object-permanence/OOD pays off) is untrained → new backlog P0; (b) **our epistemic σ may dissipate
  over the operative K-step rollout** — if it collapses with horizon, the H11/D8 self-monitor trigger
  silently dies where anticipation matters; UWM-JEPA gives the mechanism (spectrum preservation) + the
  falsifier (blind-rollout R²-retention by horizon) — impact: H15 / H11 / D8 / D9 — https://arxiv.org/abs/2605.25313
- `[Data Engineering]` [2026-07-14] [loader/license] **Cosmos-Drive-Dreams** (`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`)
  = **CC-BY-4.0** → the one *publicly-claimable* rich AV corpus (closes the gap left by the real
  PhysicalAI-AV exclusion). RDS-HQ: 5 843 clips + 81 802 synth videos, 7 weathers, 30 fps, per-frame
  4×4 `vehicle_pose`; front_wide_120fov = same 120° HFOV as PhysicalAI (D-016 focal reuse). Loader
  ships (intake pkg, 9 tests): derives steer/accel from geometry (`κ=yaw_rate/v`, low-speed clip),
  D-015 9-ch, `CORPUS_META` byte-identical to comma2k19 (D-017 I7 → admissible in the D-010 mix) —
  impact: D-014/D-002/H7/H4 — `2026-07-14-cosmos-drive-dreams-loader-and-landscape.md`
- `[Data Engineering]` [2026-07-14] [doc] `DATASET_LANDSCAPE.md` created (D-012 standing duty, was missing): 3 tiers, per-corpus
  license class / size / actions / urban-richness / cost-to-first-batch. Firewall: public numbers =
  comma2k19 + Cosmos-DD only. Next: verify WorldModel-Synthetic-Scenarios card; add Zenseact ZOD (real-CAN
  #2, H4 arm-B) — impact: D-012/G-D1 — `DATASET_LANDSCAPE.md`
- `[Data Engineering]` [2026-07-14] [arXiv] H7 latent-action/IDM surge: **LAWM** (2509.18428, latent actions from unlabeled
  video via world modeling → the labeled-bridge our comma2k19 IDM serves), **Drive-JEPA** (2601.22032,
  V-JEPA latent WM for E2E driving → "world model" no longer differentiates; moat = hierarchy+efficiency+
  imagination+self-monitoring), **HiLAM** (2603.05815, hierarchy×latent-action), **CLAW**/**DeFI**
  (label-free forward/inverse dynamics → flow/forward-consistency term for the IDM). External support
  only, no status upgrade (P8) — impact: H7/H3/H1 — same note §5
- `[Architecture & Inference]` [2026-07-14] [arXiv 2512.24497] JEPA-WM planning ablation: faithful unroll ≠ planning success —
  **decode/probe quality is necessary but NOT sufficient**. So D1–D3 are instrument gates; closed-loop
  D4–D6 arbitrate. Also: AdaLN action-cond wins (our FiLM confirmed), +RoPE best; multistep rollout loss
  = data-aug vs compounding error (2-step sim / 6-step real); ViT-L enc + depth-12 pred optimal for
  complex real dynamics (validates base250); DINO > V-JEPA encoders (H4 arm-B data point) — impact:
  D1–D3 / H1 / H4 / H5 — https://arxiv.org/abs/2512.24497
- `[Architecture & Inference]` [2026-07-14] [Meta V-JEPA 2 AC] 300 M block-causal action-conditioned latent WM predicting next-frame
  representation — same family & envelope as our 261 M operative path (D-008 scale sanity) — impact:
  H1 — https://arxiv.org/html/2506.09985v1
- `[Architecture & Inference]` [2026-07-14] [LeWM] stable end-to-end action-conditioned JEPA, 2 loss terms, no EMA/stop-grad, no
  collapse — supports our LeJEPA/SIGReg-only anti-collapse (H3, D-003); field converging on
  regularize-don't-stopgrad — impact: H3 — https://medium.com/@adnanmasood/leworldmodel-and-the-case-for-stable-latent-world-models-0e4c33ca0f3c
- `[Architecture & Inference]` [2026-07-14] [DriveMoE CVPR2026 / GEMINUS] Vision-MoE routes camera VIEWS + skill Action-MoE, on a
  LEARNED scene router. Our differentiator (H15↔H2): route the tactical/sensor MoE on ImaginationField
  epistemic σ (gate a sensor/expert only where imagination uncertainty is low) — principled, not
  black-box — impact: H2 / H8 / WP4 — https://arxiv.org/abs/2505.16278 · https://arxiv.org/abs/2507.14456
- `[Architecture & Inference]` [2026-07-14] [SqueezeBits / ModelOpt] Native TensorRT ViT INT8 is a trap (MHA/RoPE block kernel
  fusion). Use OwLite (30 % latency, 0.7 % acc drop) / DFQ-ViT / ModelOpt PTQ instead. Batch-free
  LayerNorm (our I2 choice) enables TRT-LLM fused reduce-norm on the batch-1 streaming path → keep
  LayerNorm-only + static [6,256,256] input — impact: H5 / CNCE / deploy (ESTIMATE, no measured latency)
  — https://blog.squeezebits.com/how-to-quantize-transformerbased-model-for-tensorrt-deployment-55802
- `[Architecture & Inference]` [2026-07-14] [ReflectDrive-2] masked-discrete-diffusion trajectory planners allow revision but are
  heavier than our discrete tactical vocabulary + imagine-and-select (K batched passes, ms, no CEM/
  diffusion) → Phase-1 comparison target, not adoption — impact: H5 / WP4 — https://arxiv.org/html/2605.04647v1
- `[Architecture & Inference]` [2026-07-14] [repo/theory 2606.27014] `p0-spectral-sizing` tool (backlog #0, L2): fits the
  action-conditioned transition operator (z_t,a_t)→z_{t+1}, reports σ decay / entropy effective-rank
  (offline twin of live erank) / 99%-energy knee / trade-off-optimal k* / spectral tail, and an
  OVER-/UNDER-provisioning verdict vs the 2048 readout (D-008). Recovers a known rank-5-in-32 spectrum;
  real sizing awaits a TRAINED comma2k19 checkpoint (untrained latents degenerate, P8). 8 tests —
  impact: H3 / WP3 / D-008 — `../Implementation/incoming/2026-07-14-spectral-sizing-p0/`
- `[Architecture & Inference]` [2026-07-14] [repo] D1–D3 gate runner intake pkg: instrument-doctrine gating (BLOCKED vs FAIL),
  ADE/FDE, I3 episode split, D1 vs-global-pool & D3 probe_real/imag ablations, extra_metrics seam for
  Thursday's suite; 13 tests — impact: WP6 / D-004 —
  `../Implementation/incoming/2026-07-14-gate-runner-d1-d3/`
- `[Tools&DevEnv]` [2026-07-13] [built] MetaDrive front-camera RGB path unblocks the D-010 sim arm: sim episodes were
  `[T,1,64,64]` BEV, real (comma2k19) is `[T,6,256,256]`; `MixedWindowDataset._check_contract` rejected
  the mismatch → sim arm was structurally dead. New intake pkg renders 6ch/256 2-frame RGB stacks
  (comma2k19-identical geometry/alignment) + perturbation policy + occluder/blocked-route scenarios;
  17 tests pass, 0 new deps, import 1.38 s — impact: D-010/WP2/A8 — `2026-07-13-metadrive-frontcam-rgb-and-perturbation.md`
- `[Tools&DevEnv]` [2026-07-13] [api] MetaDrive front camera: `image_observation=True`, `sensors={"rgb_camera":(RGBCamera,W,H)}`,
  `vehicle_config.image_source="rgb_camera"`; `obs["image"]` = `(H,W,3,stack_size)`, newest at `[...,-1]`,
  [0,1] float32; `image_on_cuda`≈10× (pod, VRAM). Caveat: BGR/row-flip on some backends — verify vs PNG —
  impact: WP2 — [sensors](https://metadrive-simulator.readthedocs.io/en/latest/sensors.html)
- `[Tools&DevEnv]` [2026-07-13] [opponent] NVIDIA Alpamayo 2 Super (GTC Taipei 2026-06-01): 32 B open VLA reasoning model,
  closed-loop-trained via AlpaGym on AlpaSim; GitHub/HF "this summer". ~100–300× our envelope → reinforces
  the prove-the-mechanism-not-scale thesis (C2/P5). OmniDreams = photorealistic closed-loop world-model sim
  (AlpaSim+Omniverse NuRec) — Phase-1 watch, not Phase-0 adopt — impact: P5/H15/opponent —
  [newsroom](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis)
- `[Benchmarks & Eval]` [2026-07-11] [sweep / OpenReview nG35q8pNL9] *"What Truly Matters in Trajectory Prediction for AD?"* —
  reinforces that displacement-error (ADE/FDE) on curated sets does not track what matters for driving;
  external support for our decode-gates-are-weak-claims stance and the R1 mean±CI discipline. Bootstrap-CI
  on ADE/FDE is still **rare in the field** → our power-audit rigor is a differentiator, not overhead —
  impact: validation strategy / D1 — https://openreview.net/forum?id=nG35q8pNL9
- `[Benchmarks & Eval]` [2026-07-11] [sweep / NAVSIM GH+2506.04218] **No NAVSIM-v2 leaderboard delta since 2026-07-09** (PDM-Closed
  EPDMS still 51.3 navhard; EPDMS extended-comfort compares subsequent-frame trajectories = our TMS analogue).
  No LEADERBOARD competitor-row refresh due this run — impact: LEADERBOARD currency check — https://github.com/autonomousvision/navsim
- `[Benchmarks & Eval]` [2026-07-11] [this run / power audit] **D1 ADE@1s is NOT decision-grade at the val sizes we run.** Measured
  the estimator's sampling variance on the real step-6500 ckpt + comma2k19 val (RTX 4060, $0): per-route
  ADE@1s spans **2.31–18.75 m** (CoV 0.58); the shipped single-seed `run_d1` swings **7.28 m across split
  seeds** at 4 val eps (5.46 m at 8); fixed-probe bootstrap 95 % CI half-width ±4.51 m (n=4) / ±3.13 (n=9) /
  ±2.11 (n=20). Falsifier band (½ the reported 5.18→11.52 swing) = 3.17 m → **the step-21k D1 "regression"
  is inside the estimator's own noise band** (11.52 m sits in the n=4 CI upper bound 13.55 m). Even the n=9
  step-14k read is marginal. → **Rule R1: D1/D3 open-loop gates report mean±CI over ≥5 seeds; single-seed
  points deprecated for "gate movement". Decision-grade D1 needs ≥20 val eps.** — impact: validation
  strategy / D1 / D3 integrity — `../Implementation/d1_power_audit/`, `2026-07-11-d1-ade-statistical-power-audit.md`
- `[Benchmarks & Eval]` [2026-07-11] [this run / audit] **`d1_probe_capacity.py` (loop's D1 discriminator, `0284a5c`) shares the
  small-sample fragility** — uses ~6 val eps/corpus, single-split, compares ckpt-to-ckpt ADE deltas that at
  n≈6 are <3 m CI-noise; also mixes corpora (comma direct_k1 12.11 vs physicalai 6.88 m) in the split. Its
  "info-lost vs less-linear" verdict is not decision-grade as written → recommend bootstrap + per-corpus +
  MLP-convergence check (feedback to loop; no stack edit) — impact: D1 methodology — `../Implementation/incoming/2026-07-11-d1-gate-bootstrap/`
- `[Tools&DevEnv]` [2026-07-09] [root-cause] CARLA camera-rendering on pod2 (GIPA/vulkaninfo NULL) = TWO stacked
  host-level causes: (1) RunPod pods launch `NVIDIA_DRIVER_CAPABILITIES=compute,utility` → no Vulkan
  ICD / EGL device in-container (nvidia-smi works, vulkaninfo NULL) — set by NVIDIA Container Toolkit
  at creation, unchangeable in a running container; (2) UE4.24 can't render Vulkan offscreen (Epic
  bug) → needs OpenGL or an X server. Turnkey fix = pod template with `NVIDIA_DRIVER_CAPABILITIES=all`
  (must incl. `graphics`), gate on `vulkaninfo | grep deviceName` BEFORE installing CARLA, then Xvfb
  `:99` + `CarlaUE4.sh -RenderOffScreen`. NOT urgent (milestone 1 needs no pixels) — impact: D-014/Phase-B —
  `2026-07-09-carla-render-blocker-and-testsuite-io-cost.md` §1
- `[Tools&DevEnv]` [2026-07-09] [measured] Test-suite G-E cost is dominated by **Google-Drive hydration latency**, not
  compute: cold 40.6 s vs warm 10.7 s (same 181-pass suite; reported test time 9.2 s; stack src is only
  0.44 MB / 87 files). Fix = pin `stack/` to Drive "Available offline" → cold≈warm, ~30 s saved per cold
  agent run (all 6 weekly agents), zero code. Regression-guard shipped (`profile_testsuite.py check`) —
  impact: G-E/CI/backlog#3 — `2026-07-09-carla-render-blocker-and-testsuite-io-cost.md` §2–3
- `[Tools&DevEnv]` [2026-07-09] [tooling] AlpaSim is now a PUBLIC GitHub repo (`NVlabs/alpasim`) + AlpaGym closed-loop RL;
  Alpamayo-2 Super 32 B inference/weights "this summer". Moves from announced→clonable, but still
  40–60 GB VRAM/Docker/HF-gated → verdict unchanged: **Phase-1 cloud, not Phase-0** (P5); watch for a
  lighter reference policy to seed our closed-loop harness — impact: P5/H1/opponent —
  [NVlabs/alpasim](https://github.com/NVlabs/alpasim)
- `[Data Engineering]` [2026-07-09] [measured] **PhysicalAI-AV R1 selection from cached egomotion**: 30 cached chunks → 2,850
  clips scored (0 errors), **1,926 pass the driving gate (67.6 %)** → R1=2,000 is 74 short of reachable
  from cache (needs ~1–2 more egomotion chunks). Gate failures (924) are **all speed-band**. Camera fetch
  = same 30 chunks as R0 (~60 GB) but **3.85× the clips for identical bandwidth** (per-chunk cost) →
  fetch-plan rule: extract ALL gate-passing clips per downloaded chunk. Episode-contract PASS on a real
  clip (`[199,9,256,256]` u8, 6.5 s/clip). Tool = intake pkg `2026-07-09-physicalai-r1-selection/` (3 tests)
  — impact: H7/H4/DATASET_LANDSCAPE rank #1 — `2026-07-09-physicalai-r1-selection-and-worldmodel-scenarios-license.md`
- `[Data Engineering]` [2026-07-09] [license/loader] **PhysicalAI-WorldModel-Synthetic-Scenarios** (`nvidia/…-Synthetic-Autonomous-Driving-Scenarios`)
  = **OpenMDW-1.1, UNGATED** (Linux Foundation permissive; NVIDIA's Cosmos/Nemotron license) → *preliminarily
  public-claimable* (proposed D-022; firewall held to comma+Cosmos until Sayed/legal confirms). 264 k clips /
  8.3 TB / 4K@24 fps; families cut-in 32.9 % · veh–ped 21.1 % · lanechange 12.9 % · ped 12.4 % · **weather-deg
  9.2 %** · nudging 8.8 % · **emergency-veh 2.7 %**. **⚠ card lists RGB+captions+metadata but NO ego pose/actions**
  → the "near-zero cosmos-mirror" assumption is at risk; confirm a pose field before loader work (else IDM/H7 or
  video-only). Advances SC-02/05/06 data rows — impact: D-014/H6/H15/D9/H4 — same note §2
- `[Benchmarks & Eval]` [2026-07-09] [this run / audit] **LAL-v1 is blind to smooth anticipation** — first-live SC-01 CARLA run
  scored LAL-v1 −0.7 for BOTH policies; reproduced the cliff exactly at the −1.5 m/s³ jerk trigger (a
  comfort-bounded ease-off, |jerk|<~2, never fires it). Shipped **LAL-v2** (deceleration-onset by speed
  drop; the pre-line-of-sight generalization of TTB/TTC) → +0.3…+3.1 s anticipation lead vs −0.3 s
  reactive, 7 analytic tests — impact: WP6 / G0.6 / H15 — `../Implementation/incoming/2026-07-09-lal-v2-anticipation/`
- `[Benchmarks & Eval]` [2026-07-09] [this run / audit] **SC-01 LOPS 0.834 recompute:** matches analytic E=0.8325 of the injected
  σ=0.3 noise model (inside 95% CI, N=5000, all n_occ) → reproducible, NOT seed-luck; but reactive's 0.0 is
  *structural* → proves latent-track presence not quality; reflects injected noise, not our model (P8) —
  impact: LEADERBOARD SC-01 block / H15 honesty — `audit_results.json`
- `[Benchmarks & Eval]` [2026-07-09] [arXiv 2605.09701 / 2606.07170] **NAVSIM-v2 navhard leaderboard moved (Apr 2026):**
  DriveFuture **55.5 EPDMS** (#1 learned, future-aware latent WM); DrivoR **56.3** (test-time trajectory
  opt) — both above PDM-Closed 51.3. EPDMS adds compliance sub-metrics DDC/TLC/LK + HC/EC comfort split
  (= our H9 analogue) — impact: LEADERBOARD open-loop refresh / H9 — https://arxiv.org/html/2605.09701v1
- `[Benchmarks & Eval]` [2026-07-09] [Euro-NCAP AEB / S0001457522002329] **TTB/TTC require a detectable hazard**; occlusion-AEB
  studies recommend *longer* activation thresholds under occlusion. LAL(-v2) credits braking *before*
  line-of-sight — the gap TTB structurally cannot score → grounds our anticipation metric in accepted
  metrology — impact: LAL-v2 justification / metric-gap thesis — https://www.sciencedirect.com/science/article/abs/pii/S0001457522002329
- `[Architecture & Inference]` [2026-07-09] [repo/measured] **K-step rollout bake-off — first measured arm** (backlog P0 #2; matched
  compute, 4060, 2×2000 steps, real comma2k19, 11.74 M reduced-but-real probe). K=2 vs K=1, OFAT-verified
  (`lever_diff==["train.rollout_k"]`). **(1) rollout ≈ free: +0.5 % wall-clock (749.4 vs 745.4 s), 0 extra
  params.** **(2) D2 P1 direction-acc SATURATED at 1.0 both arms** (probe fit ≈0.9999) → the backlog
  falsifier metric is ceiling-limited, NOT discriminative. **(3) discriminative signal = `imag_rel`: K=2
  cuts 1-step latent-pred error vs persistence 2.914→1.049 (−64 %) but does NOT help the 4-step horizon
  (I4 1.451→1.645, worse)** → K must cover the decode horizon (K≈4 for the 2-s D3 claim; 2512.24497 Pareto).
  D1 FAIL + D3 BLOCKED (I4>1) both → **no decision-grade claim** (D-004); decision-grade = operative-scale
  K∈{1,2,4} sweep from pod2 step-8k. No collapse (erank ~40 both) — impact: H5 / WP3 / D-018 —
  `../Research/2026-07-09-kstep-rollout-bakeoff-and-lejepa-identifiability.md` + `../Implementation/kstep_bakeoff_probe/`
- `[Architecture & Inference]` [2026-07-09] [arXiv 2605.26379] **When Does LeJEPA Learn a World Model? (LeCun/Klindt)** — LeJEPA
  (alignment + Gaussian reg = our SIGReg) **linearly & orthogonally identifies** world latents under
  stationary additive-noise transitions; **Gaussian is the UNIQUE prior** for which it holds; **"linear,
  orthogonal identifiability enables OPTIMAL latent-space planning"**; degrades gracefully; non-Gaussian
  breaks it. Translations: (a) grounds `p0-spectral-sizing`'s LINEAR transition proxy (why fit R²≈0.99–0.999)
  → D-021 sizing-to-the-knee is principled, not convenient; (b) upgrades H3 SIGReg-only anti-collapse from
  "empirically stable" (LeWM) to "provably identifiable → optimal planning" — the Epps–Pulley isotropic-
  Gaussian target IS the theorem's unique-prior condition; (c) named experiment: add an **orthogonality
  instrument** to `spectral.py` (readout covariance ~isotropic?) — makes the theorem falsifiable on our
  ckpt — impact: H3 / D-021 / D-008 — https://arxiv.org/abs/2605.26379
- `[Architecture & Inference]` [2026-07-09] [arXiv 2605.08567] ACWM action-conditioning ablation: **cross-attention beats AdaLN only for
  HIGH-dim action spaces; NO benefit for LOW-dim actions**; AdaLN (summed timestep+compressed-action
  modulation) is the standard low-cost injection. Our action space is **2-D (steer, accel) = low-dim** →
  keep AdaLN as the `adaln_conditioning` target (AdaLN>FiLM still holds) but **expected Δ is bounded**, and
  there is no reason to reach for cross-attention. Lowers my prior that the AdaLN lever clears the +2 %
  smoke bar (backlog P1 #3) — impact: H1 / H12 (adaln_conditioning planned lever) — https://arxiv.org/abs/2605.08567
- `[Architecture & Inference]` [2026-07-08] [repo/measured] **Spectral-sizing run #1 on a TRAINED ckpt (step-6500, 4060, 24 val eps,
  7,176 pairs):** fit R²=0.990 (linear proxy valid), operator effective rank ≈43, energy knee=31, k*=21 →
  **OVER-PROVISIONED**: the 2048 readout ≫ the ~tens-dim task-relevant transition rank. Rank still climbing
  (35→43 over steps 3k→6.5k) → re-measure at final Stage-0 ckpt; decision-grade evidence for **D-021** (keep
  2048 for now, keep measuring). No change executed (D-004/D-018) — impact: H3 / D-008 / D-021 —
  `Research/2026-07-08-spectral_step6500.json`
- `[Architecture & Inference]` [2026-07-08] [repo] Bake-off harness (WP3, backlog #2): OFAT one-lever-per-run driver — every variant is
  the base config with EXACTLY one field flipped (verified by a recursive dataclass `lever_diff`; a lever
  that lies about its fields raises), scored through the D1–D3 gate runner so a BLOCKED gate yields NO
  claim; multi-seed mean±95% CI; measured-params column (FLOPs/latency deferred to backlog #5, never mixed —
  G-AI2). 8 config-native runnable levers + 4 `planned` levers (AdaLN, RoPE, K-step, tactical-MoE-on-σ) that
  carry gate+hypothesis+WP pointer and refuse to run until model code lands. 16 tests; end-to-end on real
  smoke `WorldModel` → D3 BLOCKED / D2 MIXED on untrained latents (doctrine fires). Decision-grade sweep
  awaits trained ckpt — impact: WP3 / D-004 / H4·H5·H1·H15 — `../Implementation/incoming/2026-07-08-bakeoff-harness/`
- `[Architecture & Inference]` [2026-07-08] [arXiv 2606.31232] Delta-JEPA: reconstruction-free action-conditioned WM with a Latent
  Difference Action Decoder — reconstructs the executed action from the LATENT DISPLACEMENT between
  consecutive observations (= our A4 residual + A5 inverse-dynamics, arrived at independently); improves
  planning over JEPA/repr-learning baselines on 4 continuous-control tasks. Secondary summaries: AdaLN
  action injection, 6-layer causal predictor (not in abstract — flagged) — impact: H4 / H5 (residual +
  change-weight levers) — https://arxiv.org/abs/2606.31232v1
- `[Architecture & Inference]` [2026-07-08] [lit] Action-conditioned latent-predictor conditioning triangulated across 3 sources
  (2512.24497 AdaLN>FiLM +RoPE best; Delta-JEPA; OmniDreams 2606.03159 RoPE+AdaLN) + K-step rollout Pareto
  ≈ K=4 (2nd data point to 2512.24497's 6-step real). All entered as `planned` bake-off levers; each is a
  D-018 Tactic → escalate before touching the trained config — impact: H1 / H2 / H5 — see bake-off note
- `[Architecture & Inference]` [2026-07-08] [arXiv 2606.09311] FF-JEPA: hierarchical latent planners decompose long-horizon planning to
  beat compounding error + flat-CEM cost — Phase-1 comparison target; reinforces hierarchy (H1) as the
  compounding-error answer over flat rollout — impact: H1 (Phase-1 watch) — https://arxiv.org/html/2606.09311v1
- `[Data Engineering]` [2026-07-07] [measured] comma2k19 loader (D-009) decode path validated on REAL bytes: `av` decodes
  real HEVC → [200,3,256,256] uint8 @ ~105 fps (py3.13/Win), stack→[199,6,256,256] — impact: D-009/H7 —
  see `2026-07-07-comma2k19-data-card.md` §5
- `[Data Engineering]` [2026-07-07] [license/D-002] PhysicalAI-AV **real** sets (-Vehicles/-NCore/-NuRec) = NVIDIA AV Dataset
  License: **internal-dev-only, confidential, 12-month expiry, no public claims** → EXCLUDED from public
  benchmarks (comma2k19/MIT stays the public corpus). Cosmos-Drive-Dreams = **CC-BY-4.0**, ungated → the
  one publicly-claimable AV asset. Internal use needs Sayed+NVIDIA-legal sign-off ("using NVIDIA tech")
  — impact: D-002, all public claims — `2026-07-07-physicalai-av-license-review.md`
- `[Data Engineering]` [2026-07-07] [tool+finding] A8 statistics harness shipped (`stack/tanitad/data/stats.py`, 6 tests):
  per-corpus/per-domain `frame_change_fraction` distribution for change-weighting + D-010 mix. Measured
  toy=0.046 (threshold-INsensitive, hard-edged) vs comma-real=0.053→0.012 @0.05→0.10 (threshold-sensitive:
  real change is mostly small gradient, ~1.2 % large) — change-weight the small-but-real residuals —
  impact: H3/A8, D-009, D-010 — `...-validation-and-h7.md` §4a
- `[Data Engineering]` [2026-07-07] [measured] A8 on REAL highway camera `frame_change_fraction`≈0.053@0.05 / 0.012@0.10 — only
  ~1.7× the toy floor; raw-RGB under-reads consequence on low-texture highway → change-weighted loss
  justified — impact: H3/A8, W2 bake-off — `2026-07-07-comma2k19-validation-and-h7.md` §4
- `[Data Engineering]` [2026-07-07] [measured] comma2k19 = **MIT**, ungated HF `commaai/comma2k19`, ~100 GB (10×8.73 GB chunks);
  real CAN steering (steering-WHEEL deg, ratio ~15.3) + GNSS poses, zero labels; first real batch ≈1–2
  engineer-h, 0 new code — impact: D-009/H7/G-D1 — data card
- `[Data Engineering]` [2026-07-07] [finding] Windows `|` path bug: comma2k19 route dirs (`dongle|date`) illegal on Win32 →
  extract/train on Linux A40 pod only; do NOT extract Chunk zips on the Windows dev box — impact: ops —
  data card §6
- `[Data Engineering]` [2026-07-07] [arXiv] H7 deltas: LAOF (optical-flow-consistent latent actions, label-sparse gains) +
  Sensorimotor World Models (IDM-as-perception) → add flow-consistency term to seed IDM; log
  steering-ratio calibration residual — impact: H7 — `...-validation-and-h7.md` §3
- `[Tools&DevEnv]` [2026-07-06] [measured] Orin DLA does NOT support ViT attention (GPU-only, JetPack 6.2); INT8 on small
  ViTs can regress latency 2.7× (ViT-S+DPT, Orin Nano). Plan ONNX→TensorRT FP16 static-shape first, INT8
  only with measured calibration — impact: C1/P5 — see `2026-07-06-metadrive-adoption-and-alpasim-verdict.md` §5
- `[Tools&DevEnv]` [2026-07-06] [pick] Rerun.io (MIT/Apache, `pip install rerun-sdk`) = ROS-free, PyTorch-native replay/viz;
  Arrow-columnar `.rrd`, published nuScenes AV example. Chosen for backlog #2 (episode→overlay) — impact:
  WP-viz/D3/H5 — [rerun.io](https://rerun.io/docs/overview/what-is-rerun)
- `[Tools&DevEnv]` [2026-07-06] [verdict] AlpaSim/AlpaGym (NVIDIA Alpamayo, Apache-2.0, Jan 2026) = closed-loop microservice
  sim + closed-loop RL. Driver models need ~40–60 GB VRAM, Docker/SLURM + HF-token gated → NO-GO on RTX
  4060; Phase-1 cloud target for self-play with OUR <100 M driver — impact: C1/H1/H11, P5 — see note §3
- `[Tools&DevEnv]` [2026-07-06] [verdict] MetaDrive install: PyPI `metadrive-simulator` NO-GO on py3.13 (pins unbuildable
  gym 0.19); native blocker cleared (panda3d 1.10.16 + gymnasium 1.3.0 have cp313 wheels, <1 min). GO path =
  source install (GitHub main, gymnasium) in a supervised session — impact: WP2 — see note §2
- `[Architecture & Inference+Benchmarks & Eval+Data Engineering+Opponent Analyzer+Project Steering+Tools&DevEnv]` [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`
- `[Opponent Analyzer]` [2026-08-02 · Wayve deep-dive] [Wayve, primary sources] **FACT — ★ LA-Pose IS OUR H7, PUBLISHED, AT
  10.2 MILLION CLIPS** (arXiv **2604.27448**, 2026-04-30, Wayve): an **inverse-dynamics model** on
  **10.2 M unlabelled driving clips** learns **latent actions** (never told speed or heading) that
  cluster into straight/left/right/stopped **with zero pose labels**; a light head reads camera pose
  **including field-of-view and metric scale** in one forward pass; **>10 % over feed-forward SOTA** on
  Waymo + **PandaSet (unseen → zero-shot)**; **a 50-d latent bottleneck beat a 1,536-d one despite worse
  video reconstruction**; degrades in reverse motion — impact: **H7's premise is externally VALIDATED
  but H7 is dead as a differentiator in the form "IDM gives 1000× data"** (they published it first at 5
  orders more data) ⇒ **restate H7 as the data-efficiency SLOPE at matched params, which Wayve has NOT
  published**. Also directly closes our IDM pilot's two recorded gaps (**unknown intrinsics, metric
  scale**) and benchmarks on **PandaSet, for which we already hold an (unverdicted) loader** — and the
  **50-d result is third-party support for our own latent thesis (H3)** — https://arxiv.org/abs/2604.27448
- `[Opponent Analyzer]` [2026-08-02 · Wayve deep-dive] [Wayve, primary source] **FACT — the inference we carried since run #1
  is now an AFFIRMATIVE STATEMENT: GAIA-3 is "designed for offline evaluation and safety validation, NOT
  real-time in-vehicle deployment."** 15 B latent diffusion, **2 Dec 2025**; tokenizer 2× GAIA-2's, **5×
  compute, ~10× data, 9 countries / 3 continents**; conditioned on ego action, agents' 3D boxes,
  weather/time-of-day, road attributes; **"World-on-Rails" perturbation** (move ego, hold the scene);
  claims **"synthetic-test rejection rates reduced fivefold"** and that it **"reliably predicts relative
  policy performance"** — ⚠️ **with no paper, only a blog post and press release** — impact: the honest
  framing is **"their world model VALIDATES; ours DRIVES"** (stop saying they *failed* to put it in the
  loop — they chose not to); attack the ranking claim on **reviewability**, not relevance — https://wayve.ai/thinking/gaia-3/
- `[Opponent Analyzer]` [2026-08-02 · Wayve deep-dive] [Wayve] **FACT — they already have the closed loop we are blocked on.**
  **Ghost Gym** (Dec'23) = closed-loop **neural** simulator (neural renderer + simulated robot car +
  vehicle-dynamics model, **action fed back**), now powered by **PRISM-1** (Jun'24) — 4D photorealistic
  reconstruction **from cameras only, no LiDAR** — impact: **a third route around our closed-loop wall**
  (AlpaSim NO-GO + CARLA pixels host-blocked are both *renderer* problems; neural reconstruction needs
  GPUs, not a graphics-capable container). Highest-value architectural steal in this note — https://wayve.ai/thinking/ghost-gym-neural-simulator/
- `[Opponent Analyzer]` [2026-08-02 · Wayve deep-dive] [Wayve] FACT/INFER — **Rig3R** (Oct'25, **NeurIPS 2025 Spotlight**):
  ViT-L encoder + ViT-L multiview decoder + a **metadata embedding layer (camera ID, timestamps, raymap
  calibration)**, heads for pointmaps / pose raymaps / rig raymaps; **+17–45 % over baselines**, biggest
  gains on **unseen camera configurations** — impact: **a published answer to our two-rig PhysicalAI
  problem** (cy≈543 vs cy≈755; geometric-centre crop ~215 px wrong for rig B) — **rig-metadata
  conditioning is cheaper than filtering a rig out** (Data Eng) — https://wayve.ai/thinking/rig3r/
- `[Opponent Analyzer]` [2026-08-02 · Wayve deep-dive] [Wayve] FACT — **multi-country generalization, their own numbers, and
  the benchmark our H7 story must beat or reframe:** UK→US needed **500 h** of incremental US data over
  8 weeks to reach UK-equivalence (100 h → "5×", 500 h → "40×"); **Germany zero-shot "3× better"** than
  the initial US deployment; a new vehicle platform "8×" after **100 h**. ⚠️ **All relative multipliers
  off an undisclosed baseline — no miles, no intervention rate, no absolute figure**; their safety page
  likewise has **no metrics, thresholds, runtime-monitor or OOD methodology** — impact: strongest
  evidence *for* the AV2.0 bet **and** the most technical exemplar of **W-11**; **H11 (self-monitoring
  with a threshold) is the widest unoccupied gap in the field** — https://wayve.ai/thinking/multi-country-generalization/
- `[Opponent Analyzer]` [2026-08-02 · Wayve deep-dive] [Opponent Analyzer] **INFER + process note — three agreeing probes were
  still wrong.** Wayve's `/science/` page, an arXiv `all:Wayve` query (**0 results**) and an arXiv author
  search all suggested Wayve had published nothing after Mar 2025; the `/thinking/category/research/`
  archive shows **Rig3R (Oct'25)** and **LA-Pose (May'26)**. Cause: **arXiv does not index the
  affiliation string**, and Wayve's recent first authors are not the founders — impact: **never make an
  opponent-publication absence claim from arXiv search alone**; go to the lab's own research archive.
  Operating Standard #2, earned again
- `[Opponent Analyzer]` [2026-08-02 · run #5] [TanitAD / eval pod] **MEASURED — SC-13 RESOLVED, and the open-loop probe is
  RETIRED.** flagship v1, 40-ep PhysicalAI val, **stride 1 → 6,444 anchors, n=44 BRAKE_FAR over 15
  episodes**. Run #4 **reproduced to three decimals** on the recoverable stride-2 subset. Speed-matched:
  held **0.736** · frozen **0.723** · blind **0.672** · shuffled **0.634** · reactive **0.455**.
  **F-A (volume) did NOT fire** — `held − reactive` **+0.281**, episode-cluster CI [+0.009, +0.562] ⇒
  run #4's positive was **not** noise. **But the survival condition FAILS:** `held − blind` and
  `held − shuffled` CIs **include 0** ⇒ **vision attribution not established.** **The decomposition is
  the finding: ≈64 % of the gap survives with the scene DESTROYED** (a real window from a *different
  episode*), ≈32 % is the correct **static** scene, ≈5 % is motion ⇒ **a static-frame + ego-kinematic
  property, NOT a rolled-forward consequence.** ⚠️ On run #4's **anchor-level** bootstrap this would
  have read "confirmed"; the **episode-cluster** estimator flips it (44 events, 15 episodes) — same
  class as the `overlapping_holdout_se` retraction, caught pre-publication — impact: **H15 open-loop
  form weakened further; the closed loop is now the only remaining test**; the `D = CV_fwd − pred_fwd`
  monitor recommendation **survives with a rewritten rationale** (real vs a naive decel floor, but not
  vision-driven, and unproven vs a plain ego feature) — see `2026-08-02-opponent-sweep-run5.md` §1,
  substrate banked at `Implementation/sc13-real-probe/results/sc13_v1_stride1_windows.pt`
- `[Opponent Analyzer]` [2026-08-02 · run #5] [TanitAD / intake] **MEASURED — the scenario pipeline is stalled, not slow.**
  `ls stack/tanitad/eval/scenarios/` contains only `work_zone_phantom.py` + `traffic_light.py`.
  **SC-04 (11 tests, run #2), SC-13 (14, run #3) and SC-06 (16, run #4) are all still in
  `Implementation/incoming/` with UNFILLED orchestrator-verdict blocks** — oldest for three runs. All
  three **re-verified green today: 41/41** (py3.13.5, numpy 2.5.1, CPU, <0.15 s each) — impact: **H6
  DoA corrected 45 % → 35 %**; the binding blocker on H6 is **intake triage, not the renderer**; and it
  re-weights my own backlog toward measurement over authoring a fifth package into a stalled queue
- `[Opponent Analyzer]` [2026-08-02 · run #5] [Momenta/KBA] **FACT — and it RETRACTS one of our own inferences.** Momenta won a
  **Germany-wide Level-4 testing approval from the KBA** (**2026-07-29**), says it is the **first Chinese
  firm** to hold one, underpinning its **Munich** robotaxi plan; **Uber increased its stake** the same
  week; it also confirmed **robovans in Suzhou** (07-27). ⛔ This **falsifies run #3's INFER** that "EU
  political resistance to Chinese key-tech" was an EU-market-access weakness for Momenta/Pony we could
  turn into a wedge — impact: **delete that framing from every deck**; root-cause class *single-source
  geopolitical INFER promoted to a market-structure conclusion* → `RETRACTION_LOG.md` — https://cnevpost.com/2026/07/29/momenta-cleared-test-robotaxis-across-germany/
- `[Opponent Analyzer]` [2026-08-02 · run #5] [Zoox/NHTSA] FACT — **the regulatory bottleneck came off while the capability gap
  stayed open.** NHTSA **granted Zoox a commercial exemption 2026-07-30** (Federal Register **07-31**) to
  **charge for rides** with **no steering wheel/pedals/driver's seat** — the **first US authorization of
  its kind for a purpose-built AV** — **up to 2,500 vehicles over two years**; **the same day** the NHTSA
  first-responder deadline **expired with no public resolution**, and six weeks after Zoox's own smoke
  recall — impact: **W-09 strategic read inverts.** Our story may **not** be "the regulator will stop
  them"; it must be "**the capability is worth more than the exemption**", proven on SC-06/SC-13 — https://fortune.com/2026/07/31/zoox-robotaxi-steering-wheel-safety-data-gap/
- `[Opponent Analyzer]` [2026-08-02 · run #5] [US Congress/NHTSA] FACT — **W-09 escalates from regulator letter to draft
  statute**: Rep. **Kevin Mullin (D-Calif.)** introduced the **"AV Emergency Response Coordination Act"**
  (week of **2026-07-28**) — first-responder protocols, **24 h hotline**, NHTSA **minimum standards**,
  and **city authority to geofence AVs during emergencies**. Same coverage adds a **second** fleet-stall
  instance (a **December power outage stranded dozens of Waymos**) — impact: **W-09 + W-10**; **INFER:**
  a *geofencing* remedy concedes the in-vehicle capability is not expected soon, which is precisely the
  gap SC-06 measures — https://techcrunch.com/2026/07/28/waymo-robotaxi-operators-face-fresh-scrutiny-over-emergency-response-failures/
- `[Opponent Analyzer]` [2026-08-02 · run #5] [arXiv/LMB Freiburg] **FACT/INFER — ★ the sharpest differentiation risk to date,
  ahead of HWM: `Orbis 2: A Hierarchical World Model for Driving`** (**2607.15898**, 2026-07-17, Mittal
  et al., **Brox** group). Two-level: **high-level predictor of coarse scene structure over extended
  horizons** + **low-level generator conditioned on it**; standard driving-WM suite + counterfactual
  steering-responsiveness. HWM was hierarchy off-driving; **this is hierarchy ON driving.** Still ours
  (INFER, abstract-level): **representation/temporal hierarchy, not planning-time — no planner selecting
  over imagined futures**; **no params, no compute figure, no self-monitoring** — impact: **H1 must never
  again be pitched as "hierarchy"; the claim is "hierarchy a planner USES, with a number attached."**
  Architecture deep-read **top priority, ahead of SGDrive** — https://arxiv.org/abs/2607.15898
- `[Opponent Analyzer]` [2026-08-02 · run #5] [arXiv] FACT/INFER — **our second moat pillar is also being published**:
  **CheckVLA** (**2607.26789**, 07-29) uses an **action-conditioned world model to verify execution at
  run time and replan on deviation** — the mechanism of **H11 self-monitoring + A9 fallback**, on
  VLA/robotics rather than driving and **without a guarantee claim**. Field scan also shows the
  literature working on our own instruments: **Temporally Centered SIGReg** (2607.26924 — *our*
  anti-collapse method), **latent-WM physical-parameter identifiability** (2607.27017 — a principled
  form of our speed-decodability probe), **ODEWorld** (2607.27924, representation collapse),
  **Temporal-Distance JEPA** (2607.25337), and on efficiency **DriftWorld** (2607.15065, 30+ fps, 17×
  faster than diffusion) + **GigaWorld-Policy-0.5** (2607.13960, **85 ms**) — impact: H11/H1 narrative;
  seam routing → Architecture (SIGReg, identifiability, ODEWorld), Production & Opt (DriftWorld,
  GigaWorld) — https://arxiv.org/abs/2607.26789
- `[Opponent Analyzer]` [2026-08-02 · run #5] [NVIDIA, primary source] FACT — **W-05 wedge re-verified OPEN, third consecutive
  run, this time from NVIDIA's own technical post rather than press:** Alpamayo 2 Super = **32 B VLM
  backbone, "3× the parameters of prior Alpamayo models"**, +360° surround, +Meta-Action, claiming
  "state-of-the-art … reasoning quality, trajectory accuracy, alignment" — **no benchmark table, no
  latency, no compute number, no Nano tier**; weights "**summer 2026**". Also **quantization scripts
  "coming soon"** and **AlpaGym** released — an open-source high-throughput **closed-loop RL** framework
  (GRPO, stated **single-GPU → multi-node**). ⚠️ **AlpaGym is NOT new to this program** — Tools & DevEnv
  logged *"AlpaSim/AlpaGym = Phase-1 cloud (40–60 GB VRAM)"* on **2026-07-06** (`PROJECT_STATE.md` §5);
  what is new is the release + the **single-GPU** claim, which **contradicts the 40–60 GB figure on
  record** and makes it re-testable on our **A40 48 GB** — impact: **CNCE wedge intact**; **INFER:**
  shipping quantization tooling with a 3× param jump concedes the deployment problem in engineering but
  not in the benchmark table; **Tools & DevEnv: re-check the VRAM figure before further spend on a
  graphics-capable host** (AlpaSim = eval-pod NO-GO, CARLA pixels host-blocked) — https://huggingface.co/blog/nvidia/nvidia-alpamayo-2
- `[Opponent Analyzer]` [2026-08-02 · run #5] [IIHS / Fortune] FACT — **new W-11: the field has no exposure denominator.** Per
  the **Insurance Institute for Highway Safety**, most AV operators **"don't report how many miles their
  autonomous vehicles drove, which makes it impossible to calculate a crash rate,"** and there is **no
  standard for which incidents must be reported**; AV firms log many more minor events than human
  drivers do — impact: **W-11 (new)**, pairs with W-07 (CA DMV retiring disengagements). This is the
  weakness where our **published denominators + pre-registered gates + episode-cluster intervals +
  retraction log** are already the counter (H0) — https://fortune.com/2026/07/31/zoox-robotaxi-steering-wheel-safety-data-gap/
- `[Opponent Analyzer]` [2026-08-02 · run #5] [Waymo/Pony, economics] FACT/CLAIM — W-06 refresh: Waymo ~**3,700** I-PACE
  (Feb'26), ~**500 k** paid rides/week (May'26, 2× Apr'25), ~**$355 M** annualized revenue (Feb'26 —
  **CLAIM**, Sacra third-party estimate, not a filing), ~**$15–17**/ride, ~**20** trips/vehicle/day.
  **Pony.ai Q2'26 is NOT out** — it reports **2026-08-18**, so no Q2 delta exists to quote — impact:
  W-06 is not only a Chinese-operator story (~$96 k/vehicle/yr gross against a multi-sensor,
  map-maintained, remote-assisted stack); re-check Pony after 08-18 — https://sacra.com/c/waymo/
- `[Opponent Analyzer]` [2026-08-07 · run #4, real 2026-07-20] [Zoox/NHTSA] FACT — **Zoox recalled 105 vehicles** (NHTSA
  notified **2026-07-08**, public **2026-07-17**): on **2026-06-20** a Las Vegas robotaxi **drove into
  thick smoke from an active fire**, **failed to recognize the smoke**, then **suddenly braked, tried
  to turn, and halted** — inside the scene — impact: **W-09 becomes CROSS-OPERATOR** (Waymo + Zoox +
  federal directive = a *class*, not a company story) and **fuses W-09 with W-04** (smoke is obscurant
  *and* emergency cue → **one shared OOD head**); drove the SC-06 authoring this run — https://www.cnbc.com/2026/07/17/amazon-zoox-recalls-robotaxi-smoke.html
- `[Opponent Analyzer]` [2026-08-07 · run #4] [Waymo/SF] FACT — **2026-07-04 San Francisco breakdown**: dozens of Waymos
  stalled in post-fireworks gridlock at the **Presidio**; **64 vehicles** retrieved by staff/tow, some
  with **depleted batteries**; **unplanned road closures** a named contributor; one **occupied** car
  **drove over a lit firework**; SF mayor demanding stricter rules — impact: **new W-10** (fleet-scale
  mission/energy/network-disruption blindness, marked **`no-counter-yet`**); **SC-08 evidence upgraded**
  from the 2022 Cruise anecdote to a fresh large-N FACT — https://sfstandard.com/2026/07/05/waymo-sf-gridlock-fourth-of-july-2026/
- `[Opponent Analyzer]` [2026-08-07 · run #4] [TanitAD / eval pod] **MEASURED (not oracle) — NEGATIVE RESULT (P8)** — first
  SC-13 test on our own checkpoint, and the **pre-registered falsifier FIRED on replication**.
  flagship-30k, future actions **withheld**, speed confound controlled two ways. **In-domain
  (PhysicalAI, 3,241 anchors, n=23 events):** braking starting **2–3 s out (outside the 2 s rollout)**
  detected at **AUROC 0.72–0.74** vs reactive floor **0.43**. **Cross-corpus (comma2k19, 8,384 anchors,
  n=45): held 0.54–0.61 ≈ vision-blind 0.55–0.61 ≈ reactive 0.55–0.59 — indistinguishable.** Confounds:
  comma2k19 is out-of-domain and **CV beats the model there** (1.302 vs 1.874 m ADE), plus it is
  highway (29.1 m/s cruise) where CV is near-unbeatable — a failed replication, not a clean refutation
  — impact: **H15 evidence moves AGAINST the open-loop anticipation claim**; **SC-13 → live-measured
  (falsifier fired)**; the oracle collision-rate contrast is now **unsupported** and must stay out of
  external narrative; next test = in-domain volume + an arm that beats CV on the target corpus — see
  `2026-08-07-opponent-sweep-w5.md` §1, archive `Implementation/sc13-real-probe/`
- `[Opponent Analyzer]` [2026-08-07 · run #4] [arXiv] FACT/INFER — **HWM, "Hierarchical Planning with Latent World Models"**
  (**2604.03208**, Zhang/Terver/Zholus et al., Apr'26 rev Jun'26): world models at **multiple temporal
  scales in one latent space**, long-horizon predictions used as **subgoals for the short-horizon model
  via latent matching**, no rewards/hierarchical policy, **up to 3× less planning compute**.
  **Planning-time hierarchy — our H1 claim — is now published**, though on **manipulation/maze, not
  driving**, with **no param count** and **no self-monitoring/OOD guarantee** — impact: H1 must be
  positioned as hierarchy+efficiency+in-loop-imagination+self-monitoring *on driving*; also the closest
  published relative of the **v3** direction (DINO-WM lineage) → **Architecture deep-read, top
  priority** — https://arxiv.org/abs/2604.03208
- `[Opponent Analyzer]` [2026-08-07 · run #4] [Opponent Analyzer] INFER (design-oracle, P8) — **Emergency-Scene** scenario
  (SC-06, W-09) shipped, **16/16 tests**: corridor **incursion rate 0.0 (yield) vs 0.2 (rule-literal)**;
  **blockage 0.0 s vs 2.54 s** (12.7 s at thick smoke); **detection lead time +5.70 s vs +2.84 s**
  (−0.10 s at thick smoke). Mechanism: the obscurant collapses **object** range **90→13.5 m** while
  **scene**-level OOD range falls only **80→68 m**. **The failure is a CLIFF not a slope** → graded
  obscurant sweeps are mandatory — impact: **H11/H15/A9**; **blocked on SC-05's D8 detector**, which is
  currently failing — see `2026-08-07-opponent-sweep-w5.md` §2
- `[Opponent Analyzer]` [2026-08-07 · run #4] [NHTSA/Wayve/Pony/NVIDIA] FACT (deltas) — the first-responder deadline is for
  **presenting fixes in meetings, NOT deployed fixes** (correction, do not overstate); **Wayve $85 M
  employee tender (07-01)** = liquidity, not new capital; **Pony** reaffirms **>3,500 robotaxis / 20+
  cities** 2026 (**W-06 unchanged** — fleet targets still outrun revenue); **NVIDIA**: Alpamayo 1 =
  **10 B** open weights, **2 Super = 32 B** "expected this summer", AlpaSim open — **still no Nano-tier
  CNCE number**, our W-05 wedge stays open — https://www.axios.com/2026/07/15/waymo-accountability-emergencies-nhtsa
- `[Opponent Analyzer]` [2026-07-31 · run #3, real 2026-07-17] [NHTSA] FACT — ODI issued a formal **ADS-developers letter
  (2026-07-08)** demanding every AV developer fix, **by end of July 2026**, a **"clear pattern"** of
  robotaxis interfering with first responders (driving into emergency scenes; blocking ambulances/fire;
  failing to recognize **flashing lights, flares, smoke, fire, cones**). ≥6 incidents through Mar 2026
  needed responders to **physically move Waymo vehicles**. Morrison: **"functional insufficiency"**;
  **"Emergency scenes are not rare or extreme edge cases"** — impact: **new W-09; SC-06 elevated →
  H15/H11/A9/H9**; strongest external endorsement of the scenario-DB thesis (H0/H6) — https://techcrunch.com/2026/07/08/feds-demand-autonomous-vehicle-companies-stop-interfering-with-first-responders/
- `[Opponent Analyzer]` [2026-07-31 · run #3] [Opponent Analyzer] INFER (design-oracle, P8) — **Stationary-Lead** scenario
  (SC-13, W-08) shipped: over the classification-ambiguity sweep {0…1}, **collision rate imagination 0.0
  / detection-reactive 0.4**; **braking-onset lead time +1.20 s vs −1.26 s**; forward model **invariant
  to ambiguity** (min-TTC 2.88 s) while reactive degrades to a collision (drops the lead ≥ 0.75); OKRI
  ~3.2× lower (**14/14 tests**) — impact: **H15/A9**, first collision-rate + lead-time contrast for the
  consequence-forward-model thesis — see `2026-07-31-opponent-sweep-w4.md`
- `[Opponent Analyzer]` [2026-07-31 · run #3] [Avride/NHTSA] FACT (delta) — PE opened **2026-05-06**; the 16 crashes span
  Dec'25–Mar'26 (**≥9 Dallas**, rest Austin), Hyundai **Ioniq 5** on Uber; NHTSA video: "unsafe lane
  changes into the path of other cars, **failing to avoid slow-moving vehicles ahead, and striking
  stationary objects**"; **all 16 under a safety monitor, only 1 attempted to intervene** — impact:
  reinforces **W-08/SC-13** (systematic, not marginal) — https://www.nbcdfw.com/news/local/robotaxi-operator-under-investigation-for-crashes-in-dallas/4023503/
- `[Opponent Analyzer]` [2026-07-31 · run #3] [Wayve/Pony/NVIDIA/Uber-field] FACT — **Wayve +$60 M** (AMD/Arm/Qualcomm) Series-D
  extension + **Tokyo pilot late'26 (Nissan LEAF)**; **Pony Q2'26 200+ Gen-7 built, rev +76%**, orders
  +119% vs Jan (net loss Q1 $50.4 M — W-06 unchanged); **NVIDIA/Mercedes CLA = MB.Drive Assist Pro** L2++
  (10cam/5radar/12us), Alpamayo pitched to solve the "long tail," **no Nano-tier CNCE number**; Uber is now
  a **multi-vendor L4 marketplace** (Waymo/Avride/Autobrains/Momenta/Wayve/Nuro/WeRide) — impact: W-05/W-06
  hold; distribution moat is Uber's → technical-moat premium rises (H0) — https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/
- `[Opponent Analyzer]` [2026-07-31 · run #3] [emerging: Zoox/WeRide/Nuro] FACT — **Zoox** production-intent robotaxi unveiled
  (Jun'26), gated on NHTSA approval for up to **2,500** no-manual-controls vehicles (regulatory, not
  capability, bottleneck); **WeRide** driverless-fare via Uber in **Dubai (Mar 31'26)** + Abu Dhabi/Riyadh
  (1,200+ by ~2027); **Nuro+Lucid+Uber** expanded to **≥35,000 Lucid** vehicles, SF-Bay later'26 — impact:
  L4 field widening; all compute-heavy multi-sensor (W-05) — https://www.cnbc.com/2026/06/24/amazons-zoox-unveils-redesigned-robotaxi-ahead-of-upcoming-expansion.html
- `[Opponent Analyzer]` [2026-07-31 · run #3] [arXiv] FACT/INFER — AD latent-WM surge continues; **hierarchy now surfacing** —
  **SGDrive** (2601.05640) "scene-to-goal *hierarchical* world cognition"; **DriveFuture** (2605.09701,
  1st NAVSIM-v2 navhard Apr'26); **EponaV2** (2605.14696); **Latent-WAM** (2603.24581) — impact: H1
  differentiator being explored (not yet with our combination); deep-read SGDrive next (Architecture) — https://arxiv.org/abs/2601.05640
