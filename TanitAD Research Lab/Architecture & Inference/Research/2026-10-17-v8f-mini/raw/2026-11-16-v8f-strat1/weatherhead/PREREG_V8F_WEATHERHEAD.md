# PREREG_V8F_WEATHERHEAD (FROZEN 2026-11-16 by the MM, NOT RUN): the attention-pooled detected-information head on the frozen STRAT1 model

Status: drafted 2026-11-16 by an Architecture agent, BEFORE any head was built, fitted or scored; FROZEN by the MM 2026-11-16 ~22:45Z with the rulings in `MM_RULINGS_WEATHERHEAD.md` (binding, same weight as this text). Nothing in this card has run. sha256 in `PREREG_V8F_WEATHERHEAD.sha256`. Bars, arms, controls and outcomes below do not change after any number is seen.
Follows: frozen STRAT1 card `PREREG_V8F_STRAT1.md` (sha256 `02c16b83...`) section 6, row NOT-DETECTED(v): *"If the probe detects, the next arm is that head"*; and `RESULT_WEATHERPROBE.md` (prereg sha256 `0582c783...`): DETECTS(weather) by the probe at both S1 seeds.
Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I, another agent or doc, not re-verified here), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U). Every detected-info number is tier **T0** (per-frame perception output at the window's NOW frame); every driving number is tier **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), as STRAT1.
Paths: `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`; `S1D` = `PK/raw/2026-11-16-v8f-strat1`.

## 0. Decisions for the MM / PI (escalated)

1. **Form (a) is a head fit on a FROZEN driving model: zero training of any driving parameter** (S, T, O, adapter, `pos`, `time`, `e_*`, the shipped `ctx_head`). It fits a new 1.33 M-parameter output head on cached features, the same class of computation as the zero-training probe the frozen STRAT1 card already prescribed, but its product is a new shipped model output. `RESULT_WEATHERPROBE.md` wrote "a new card, a PI go for training". **MM ruling needed:** whether "Build the arm" (PI, 2026-11-16) plus the retrain authorisation of 2026-11-08 (I, register row `PI: horizon bands`) covers a frozen-model head fit with no pod time; if not, `QUESTIONS_WEATHERHEAD.md` Q4 goes to the PI. Recommendation: covered (no driving parameter moves, no pod or box-2 time, the driving outputs are bit-identical by test).
2. **Gate route.** A study checker `wh_check.py` that mints `wh-gate-token-v1` (section 4.6), as the STRAT1, OBD and ACTSENS study tokens. The PI rule of 2026-09-26 speaks of "a training session"; a head fit is one. Recommendation: require the token (cheap; the probe ran without one and this card ships a model output).
3. **Fit cache: the dev-box train cache** `F:/TanitAD-scratch/speedread/cache/train` (4,572 clips, pipeline `ra1a-pooled-v1`, grids 4x10 and 8x20, M: manifest read 2026-11-16). It is the same encode pipeline as the eval cache of record `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval` (141 clips, `ra1a-pooled-v1`, `dinov3-vitl16-pretrain-lvd1689m`, M: its `meta.json`). Whether it equals the pod train cache S1 trained on is **U**: the pod's eval cache is a separate re-encode (`D-EVALCACHE-POD-DEVBOX`, I). Precondition P-CACHE (section 4.5) prints the content_md5 match count; no bar depends on it.
4. **All five variables, one fit; the output source is chosen per variable by a rule committed here** (section 3.3). Recommendation: accept.
5. **Base commit**: `a4357e79` (`agent/v8f-strat1-20261116` tip; the read code of record `e81bde01` is its ancestor, M `git merge-base`). The new flag defaults off, and off is `a4357e79` bit for bit (snapshot test).
6. **PI decisions** (`QUESTIONS_WEATHERHEAD.md`): Q1 conditional pod go for the fallback form (b); Q2 whether "readable from the model's vision input features" is an acceptable detected-weather deliverable; Q3 PI frames on the minority weather classes (the label lever); Q4 only if the MM rules item 1 not covered.

## 1. Question

Does an attention-pooled head over the 4 history frames x 40 adapted tokens, fitted on S1's FROZEN model and shipped as a model output with its own checkpoint key, detect weather (and the other four STRAT1 variables) better than a constant at both S1 seeds, with every planner output bit-identical to S1, and does S1's trained adapter make the information more readable than a random projection of the same cached features?

## 2. What is known before this card

| fact | value | class, artifact |
|---|---|---|
| STRAT1 shipped head, weather (token-mean of NOW, MLP) | s0 0.9037, +0.0463 [-0.0108, +0.1014] not separated; s1 0.9291, +0.0717 [+0.0254, +0.1204]; balanced 0.7084 / 0.6711 | M, `S1D/read/out/STRAT1_CTX.json` |
| STRAT1 label | NOT-DETECTED(weather); country (+ BEYOND-RIG), surface, illumination, period_of_day DETECT at both seeds | M, same |
| Probe on S1's frozen adapter (attention pool, 160 tokens) | s0 0.9358, +0.0784 [+0.0225, +0.1305]; s1 0.9387, +0.0813 [+0.0283, +0.1326]; balanced 0.8050 / 0.7802 | M, `S1D/weatherprobe/weatherprobe.json` |
| Probe, labels shuffled between clips (R1) | +0.0000 [0, 0] at both seeds | M, same |
| Probe on a RANDOM adapter (R2, seed 0, E 8) | 0.9287, +0.0713 [+0.0171, +0.1247], separated | M, same |
| Probe fit cost | about 9-10 s per epoch for one 4-class probe on 36,051 windows (8 epochs 20:05:07 to 20:06:18); whole run 673 s; peak GPU 467 MiB; LoRA pod GPU | M, `wprobe_log.txt`; GPU model RTX 4000 Ada is I (memory) |
| Train windows of the probe | 36,051 from 4,572 clips, 8 NOW frames per clip; clear / fog / rain / snow 30,988 / 831 / 1,647 / 2,585 | M, `RESULT_WEATHERPROBE.md` |
| Eval constants, window basis | country 0.0921, weather 0.8574, surface 0.7970, illumination 0.6062 (2,359 windows / 140 episodes, twilight excluded), period_of_day 0.2409; country rig baseline 0.0779 | M, `STRAT1_CTX.json` |
| Twilight | 240 eval steps in 5 clips, excluded from illumination | M, `STRAT1_CTX.json` |
| Label ceilings vs the PI's own frames | weather 36/40 = 90.0 % [76.9, 96.0]; surface 32/40 = 80.0 % [65.2, 89.5]; illumination 30/39; country and period_of_day come from metadata (country, clock hour), no PI-frame row | I, register `H-CTX-LABELS-V1`, STRAT1 card section 2 |
| Planner scene assembly | `_scene(adapt, ok, last)` returns the `hist` frames ending at `last` as [B, hist x n_tok, d] tokens plus `time`, with a pad mask from `frame_ok`; `_encode` = adapter + `pos` | M, `v8f_mini.py:912-919`, `:1036-1038` at `a4357e79` |
| Shipped head | `ctx_head = _mlp(256, 256, 41)` (76,329 params) on `tok.mean(1)` of the NOW frame; built last under `CTX_INIT_SEED = 20261116`; `ctx_logits` feed the loss and the evaluator only; T-ISO = `ctx_isolation_problems` over `CTX_ISOLATED_OUTPUTS` | M, `v8f_mini.py:155-163, 897-901, 958-963, 1664-1695` at `a4357e79` |

**What the probe does and does not settle.** Weather is readable from the cached DINOv3 4x10 tokens over 4 frames by a learned attention pool. It does not show that S1's training put weather there (R2 detects too). This card turns the probe into a shipped output and measures the trained-versus-random question with a replicate, which the probe could not (one fit each).

## 3. Form: the choice, and why

### 3.1 The two forms

| | **(a) frozen-model head (RECOMMENDED)** | **(b) joint pooled ctx head (FALLBACK)** |
|---|---|---|
| what trains | a 1.33 M-param head only; every S1 tensor byte-identical | the full S1 recipe, 3,000-step fine-tune from M2b, the pooled head in place of the mean+MLP head |
| driving risk | none by construction; checked by bit-identity (T-ISO d) | the head's CE moves the shared adapter (`--ctx-detach off`), so driving needs the 24-cell read at both seeds |
| attribution | the head is the only change | confounded with STRAT1's own open driving question: S1 - C0 is SEEDS-DISAGREE and its one-seed costs are not yet assigned to the nav text or the heads (M, `RESULT_STRAT1.md`) |
| portability | re-fit in minutes on any adopted checkpoint (S1 at seeds 10/11, N1, C0) | tied to one trained arm |
| what it can add over the probe | a shipped key, a replicate, the trained-vs-random read | the adapter may become more weather-readable; the information itself is already in the features (R2) |
| compute | dev-box GPU, about 3-6 h (E); no pod time | LoRA pod, about 2.3 pod-h for 3 arms (E) plus reads; PI training go |
| gate | `wh-gate-token-v1` (section 4.6) | the full `strat1_check.py`-style gate with a new trainer flag and its G-DVB entry |

### 3.2 Recommendation

**Form (a).** It delivers the output the frozen card asked for at zero driving risk, isolates the head as the only change, decouples detected information from the unresolved nav-text adoption, and answers the trained-vs-random question with controls. Form (b) is kept as the fallback with a measured trigger (section 7, branch WH-NOT-DETECTED, FEAT row): it is the right lever only if a head with its own trainable projection reads weather where the frozen S1 adapter does not, that is, if the frozen adapter is the bottleneck.

### 3.3 All five variables or weather only

* **One fit of five independent sub-heads**, one per STRAT1 variable, each exactly the probe's architecture. Their parameters are disjoint, so each sub-head's gradient is its own CE's, and AdamW per-parameter updates make each sub-head's fit independent of the others given the window set and batch order (Adam is scale-invariant up to eps, so the shipped head's per-variable weights, surface 0.5, carry no meaning here and are not used).
* **Output source per variable, committed here:**
  * weather: from the pooled head if WH-DETECTS (section 7); the shipped head is NOT-DETECTED, so there is nothing to keep;
  * each other variable v: the pooled sub-head replaces the shipped `ctx_head` slot only under POOL-REPLACES(v) (section 7), otherwise the shipped slot stays and the pooled sub-head is reported beside.
* Why fit all five: period_of_day (0.4785 / 0.4956) and country (0.4940 / 0.5369) are the weakest absolute reads (M, `STRAT1_CTX.json`), and the PI ruled period of day and illumination are "estimated from images and past frames only" (I, `D-HWM-BANDS-1`); the pooled head is the first STRAT1 output that reads past frames. The extra cost is a few GPU-minutes.

## 4. Build of form (a) (one Sonnet builder; nothing here is built yet)

### 4.1 Module, flag, key
* `V8FMiniConfig.ctx_pool: str = "off"` (`CTX_HEAD_MODES`). Off = no module, no tensor op, no output key: `a4357e79` bit for bit (pinned snapshot test, as STRAT1's off = `cdbff971`).
* On: `self.ctx_pool_head = nn.ModuleDict({v: PoolHead(256, n_v) for v in CTX_VARS})`, built LAST under `torch.random.fork_rng` with `CTX_POOL_INIT_SEED = 20261117`, so every module above and the global stream after it are the off model's.
* `PoolHead(d, n)` = `LayerNorm(d)` -> `MultiheadAttention(d, 4 heads, batch_first)` with ONE learned query `[1, 1, d]` ~ N(0, 0.02^2), key padding mask from the scene pad -> `LayerNorm(d)` -> `Linear(d, n)`. Params 264,448 + 257 n per sub-head; 265,476 for weather (= the probe's "about 0.27 M"); **1,332,777 for the five** (computed from the architecture).
* Input: `tok, pad = self._scene(adapt, b["frame_ok"], last_now)`, the planner's own assembly of frames t-3..t (160 tokens, adapter + `pos` + `time`). Rows whose NOW frame is not real are masked in the loss and the read, as the shipped head (`fok = frame_ok[:, -1]`); a row whose 4 frames are all padded is given an unmasked NOW slot so attention never returns NaN, and is masked anyway.
* Output `out["ctx_pool_logits"]` [B, 41] in `CTX_VARS` order (the same layout as `ctx_logits`, so the STRAT1 reader's `split_logits` reads it unchanged). Checkpoint keys `ctx_pool_head.<var>.*`; `CTX_POOL_KEY_PREFIXES = ("ctx_pool_head.",)`; built levers record `ctx_pool`, `ctx_pool_dim 41`, `ctx_pool_vision_only_source`.
* In form (a) the fit computes `adapt` and the scene tokens under `torch.no_grad()`: the frozen model gets no gradient path.

### 4.2 Fit recipe (fixed now)
* Train windows: per train clip of `strat_labels_table.npz` (sha256 `8055a321...`), NOW frames t_k = round(linspace(3, n_frames - 1, 8)), unique (the probe's set). Row = `train_off[clip] + t` (the STRAT1 convention). A row enters variable v's CE when `flag == 0` and `<v>_valid`, and for illumination when the label is not twilight (`CTX_TWILIGHT_ID = 3`). The 60 unlabeled clips (`--exclude 540:600` in the arms) enter no CE.
* AdamW lr 1e-3, weight decay 0.01, 100 warm-up steps, cosine to 0, batch 256, fp32 head, features cast from float8 to fp32, unweighted CE per variable, loss = sum over variables. Fit seed s sets the head initialisation (forked RNG, `manual_seed(s)`) and the batch order (`np.random.RandomState(s)`).
* **Epochs, chosen per variable on TRAIN only:** E in {4, 8, 16} (the probe's E = 8 was the top of its grid and still rising, M). Train clips split 80 / 20 by a seed-0 clip permutation; all five sub-heads fitted on the 80 % at each E; for each v, E_v = argmax of held-out (accuracy - majority accuracy), ties to the smaller E. The final head is fitted on all train windows once at each distinct chosen E, and sub-head v is taken from the fit at E_v (legitimate because the sub-heads are independent). E_v is chosen for H_s0 and H_s1 separately and recorded; every other arm uses the E_v of its seed's H arm.
* The eval set is never loaded by the fit (disjointness section 4.5).

### 4.3 Admissibility
* **Vision only (T-VIS).** `_ctx_pool` reads `feats` (through `_encode`) and `frame_ok` only: the AST guard `CTX_FORBIDDEN_NAMES` extended to it; key-guarded read set = `CTX_ALLOWED_KEYS`; logits bit-identical when every non-vision batch key (ego, nav, limit, ages, labels) is replaced, different when `feats` is. Reading the 4 history frames meets the PI ruling "images and past frames only".
* **Goal-input rule (T-ISO).** `ctx_pool_logits` feed neither the planner nor any goal path. Four checks, all must pass:
  * (a) state-dict identity: every key of S1's checkpoint is present in the head checkpoint with a byte-identical tensor (sha256 per tensor); the only new keys start with `ctx_pool_head.`;
  * (b) perturbation: `ctx_isolation_problems` extended so that N(0, 1) added to every `ctx_pool_head` parameter leaves `CTX_ISOLATED_OUTPUTS` and `ctx_logits` bit-identical and moves `ctx_pool_logits`;
  * (c) full grid: `forward_infer` at batch 64 on all 2,399 eval windows, S1 vs S1 + head: every `CTX_ISOLATED_OUTPUTS` key and `ctx_logits` `torch.equal`;
  * (d) driving dumps: T1-kin dumps (`--limit causal`, batch 32, as the read) of S1 re-dumped and of S1 + head, in ONE environment and one thread count: every planner array byte-equal. The 24 driving cells and the 6 route-setting cells then read exactly 0.0 [0, 0]: a **structural zero** (an identity, not an estimate), labelled so.
* Detected-info outputs into a goal path are a later card with its own admissibility check.

### 4.4 Reachability (CLAUDE.md 2026-09-10/11)
1. **Trace**: `--ctx-pool on` (fit script and both evaluators) -> `V8FMiniConfig.ctx_pool` -> `V8FMini.ctx_pool_head` -> `_forward` emits `ctx_pool_logits` -> fit loss terms `T["ctx_pool/<var>"]` -> `loss.backward()`. The build report names `file:line` at each hop.
2. **Non-zero and exact-zero**: on one real batch with `requires_grad` True on EVERY model parameter, backward of the fit loss gives a non-zero gradient on each of the five sub-heads and **exactly 0.0 (or None) on every other parameter**, adapter included (the no-grad token path). The exact-zero half proves the probe can read zero.
3. **Adam-state census** after each fit: optimizer state exists for exactly the 55 head tensors (11 per sub-head) and for no other parameter (`probe_zerograd3.py` logic).
4. **Defaults**: every arm's argv names `--ctx-pool on` (the S1 re-dump names `off`). The reader refuses an H dump without `ctx_pool_logits` (n/a, reason "no output"); it never falls back to `ctx_logits`.
5. **Mutations, each must turn its named test RED**: (m1) delete the fit-loss line -> G-LIVE; (m2) drop the `no_grad` on the token path -> the exact-zero control; (m3) feed `ctx_pool_logits` into O's conditioning or `code` -> T-ISO; (m4) name an ego or nav key inside `_ctx_pool` -> T-VIS; (m5) evaluator drops `--ctx-pool` -> reader "no output"; (m6) reader substitutes `ctx_logits` for an H arm -> K-ID.

### 4.5 Preconditions (checked before the first fit; any failure stops the chain)
* **Disjointness** (must read 0, as the probe): table train vs eval clips; fit clip files vs the 141 eval clips; fit clip files vs the eval grid's `order`.
* **Pins**: S1_s0 `6c2af475...`, S1_s1 `e51b756f...`, C0_s0 `ad91be31...`, C0_s1 `17c751a0...` (M, `RESULT_STRAT1.json` provenance; dev-box copies `F:/TanitAD-scratch/strat1/read/ckpts/<arm>/ckpt_last.pt`); label table `8055a321...`; eval grid `659c3b65...`; `strat1_ctx_read.py` `1b9377ba...`; `taniteval/ci.py` `528276c5...` (read tree copy).
* **P-CACHE** (beside, no bar): content_md5 of the dev-box train manifest against the pod train manifest (`/root/data/cache_pod/train/manifest.jsonl`, a file the MM pulls; no pod job). Printed as a count of 4,572.
* Free space: F: above its 20 GB floor, C: at or above 6 GB (memory rules); host RAM for 36k windows x 160 x 1024 float8 = about 5.9 GB (E).

### 4.6 Gate: `wh-gate-token-v1`
Bound to commit, argv sha256 and every input sha256 above. Mint refuses unless each check reads PASS, and each check has a deliberate-regression arm that must make it FAIL:
* G-HYG: the config refuses an undeclared `ctx_pool_*` attribute;
* G-DVB: argv levers (`--ctx-pool`, E grid, fit seed, init checkpoint sha256, label sha256, train manifest sha256) equal the built and recorded values;
* G-LIVE: 4.4 items 2 and 3;
* G-CLOCK: the fit's row index equals `label_rows` of `strat1_ctx_read.py` on 100 random train rows; a t + 1 shift arm FAILS;
* G-EVAL: each evaluator loads the head checkpoint strictly and its `ctx_pool_logits` equal the fit script's on the eval windows to atol 1e-5;
* G-CKPT: save / load round trip, byte-equal tensors;
* G-SUITE: clean-tree tests, 0 regressions against `a4357e79`.

## 5. Arms and controls (form (a); priority order, not a dependency chain)

All arms are head fits on frozen features or eval-only passes. "E_v" = the per-variable epochs of that seed's H arm.

| P | arm | frozen adapter (`adapter`, `pos`, `time`) | head fit | role |
|---|---|---|---|---|
| 1 | **H_s0** | S1_s0 | seed 0, E_v chosen | the head arm |
| 2 | **H_s1** | S1_s1 | seed 0, E_v chosen | second training seed (the replicate pair the bar reads) |
| 3 | **RA_s0** | RANDOM: `nn.Linear` default init, LayerNorm (1, 0), `pos` / `time` N(0, 0.02^2), init seed 0 | seed 0, E_v of H_s0 | random-adapter control, MANDATORY |
| 4 | **RA_s1** | RANDOM, init seed 1 | seed 0, E_v of H_s1 | random-adapter control, second seed |
| 5 | **R1_s0, R1_s1** | S1_s0 / S1_s1 | train labels permuted between train CLIPS (seed-0 clip permutation, rows by position modulo, all five variables together) | deliberate-regression arm: must read NOT DETECTS on every variable |
| 6 | **H_s0_rep** | S1_s0 | **seed 1**, E_v of H_s0 | fit replicate: floor F_fit = abs(H_s0_rep - H_s0) per variable |
| 7 | **K4p_s0, K4p_s1** | H_s0 / H_s1 with `ctx_pool_head` re-initialised (`CTX_POOL_INIT_SEED`, forked RNG), eval only | none | must read NOT DETECTS on every variable |
| 8 | FEAT_s0, FEAT_s1 | none frozen: a fresh `LayerNorm(1024) -> Linear(1024, 256)` + `pos` + `time` (init seed 0 / 1) TRAINED with the head | seed 0 / 1, E_v of H_s0 / H_s1 | feature ceiling at this pooling; decides the failure branch (section 7) |
| 9 | C0A_s0, C0A_s1 | C0_s0 / C0_s1 (driving-trained, no ctx loss) | seed 0, E_v of H_s0 / H_s1 | beside: did S1's ctx loss change what the adapter carries |

Reader controls (no fit): **K0** anchor (this card's reader on S1's banked `ctx_logits` reproduces every `STRAT1_CTX.json` row: accuracy, constant, delta and CI bounds equal as floats); **K2** constant predictor reads exactly the constant; **K3** planted labels read 1.0 and DETECTS; **K5** eval labels clip-shuffled (seed 0) against H predictions read NOT DETECTS; **K6** rig-only country reads 0.0779 and NOT DETECTS-BEYOND-RIG; **K-ID** the evaluator's `ctx_pool_logits` equal the fit script's (atol 1e-5) on all 2,399 windows. Plus T-ISO (a)-(d), T-VIS and the reachability set of section 4.4.

A chain stopped after P2 yields the detection read at both seeds; after P4 the mandatory wording control; after P6 the floor.

## 6. Read

* **Windows, cache, labels**: the eval grid of record (2,399 windows over 141 episodes, `eval_grid_windows.json` `659c3b65...`), NOW frame t, features from the **dev-box eval cache** `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval`, the very cache of the STRAT1 T0 read. The pod's eval cache is never read and no number from it is compared with these (`D-EVALCACHE-POD-DEVBOX`). Labels by `label_rows` of `strat1_ctx_read.py` (flag != 0 and invalid masked; illumination twilight masked: n 2,359 windows / 140 episodes).
* **Per variable, per arm, per seed**: accuracy, balanced accuracy, per-class recall, confusion; country also the rig row and the within-platform strata. Constants = the window-basis constants of section 2, unchanged.
* **DETECTS(v)** = accuracy - constant > 0 and separated at BOTH S1 seeds (H_s0 and H_s1), and K4p NOT DETECTS at that seed. Country also needs DETECTS-BEYOND-RIG (accuracy - 0.0779 > 0 separated at both seeds). This is the frozen STRAT1 rule with K4 replaced by its form-(a) twin.
* **Paired differences** (same windows): H - RA (trained vs random); H - shipped (S1's `ctx_logits`); beside, no bar: H - FEAT, H - C0A, H - probe (weather, `wprobe_out.npz` rows on the dev-box eval cache).
* **Estimator**: `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, over the eval episodes (read tree copy, sha256 above). It answers "another draw of EPISODES". "Another TRAINING RUN" is answered by the two S1 seeds (two trainings, two adapters) and "another FIT" by F_fit (one replicate pair, a weak floor, stated). "Another INFERENCE RUN" does not arise: the head and the planner are deterministic and every dump runs in one environment and one thread count.
* **Label-ceiling context, printed beside every row**: weather 36/40 = 90.0 % [76.9, 96.0], surface 32/40, illumination 30/39 against the PI's frames (I); country and period_of_day: metadata labels, no PI-frame row. No number here is called accuracy against the world. A head at or above the label's own agreement with the PI cannot be told from label noise by these reads.
* **Driving families** (BINDING four families; STRATEGIC unpaused): for form (a) LONGITUDINAL, LATERAL and TACTICAL (24 cells per seed) and STRATEGIC route setting (6 cells per seed) are reported as the T-ISO (d) structural zero with the byte-equality evidence; STRATEGIC route following (S_CKPT) likewise (the `ckpt` arrays are among the byte-equal ones); STRATEGIC detected information = this read; derived goals: not in this arm (later card, goal-input rule); strategic intent 6-30 s: not computable (no output beyond 80 m of arc); visibility: not built (label 25/40 below its constant 31/40, I).

## 7. Outcomes, committed before any number (each names what runs next)

**Instrument stage, read first.** If any of K0, K2, K3, K4p, K5, K6, K-ID, R1 (either seed), T-ISO (a)-(d), T-VIS, the reachability set, disjointness or G-EVAL fails, the outcome is **INSTRUMENT-FAIL**: no number is quoted, the fault is found and fixed, the failing stage re-runs, and no bar changes. A T-ISO failure means the head reaches the planner or the dump environment differs; nothing ships until it passes.

**Weather (the primary variable)**

| outcome | condition | next |
|---|---|---|
| **WH-DETECTS** | DETECTS(weather) by H at both seeds (section 6) | weather is a reported model output from `ctx_pool_head.weather`; the wording is set by the next two rows; then (i) the label lever (Q3: PI frames on fog, rain, snow), since the head already sits at or above the label's agreement with the PI; (ii) the head is re-fitted on whichever driving checkpoint the SEEDS-DISAGREE branch adopts (minutes); (iii) the derived-goals card (weather and surface as strategic constraints; its own goal-input admissibility) |
| + **WH-ADAPTER-ADDS** | H - RA > 0 separated at both seeds AND abs(delta) > F_fit | wording: "S1's adapter makes weather more readable than a random projection of the same cached features" |
| + **WH-INPUT-FEATURES** | WH-DETECTS and not WH-ADAPTER-ADDS | wording: "weather is readable from the model's vision input features (cached DINOv3 4x10 tokens over 4 frames) by the shipped pooled head; S1's training is not shown to add to it". This is the deliverable as written; it does not by itself trigger form (b) (Q2) |
| **WH-NOT-DETECTED** | not separated at one or both seeds | **a-W** next: a weather-only sub-head with the probe recipe exactly (weather-valid windows only, E grid {2, 4, 8}), same seeds, controls and reads. a-W DETECTS -> weather ships from a-W, and the cause (window set or E grid) is named. a-W NOT-DETECTED -> the FEAT rows decide: **FEAT detects at both fit seeds** -> the frozen S1 adapter is the bottleneck -> **form (b)** (section 8; PI go Q1); **FEAT does not** -> the representation lever card: a frozen-feature pooled head on `pool8x20` (160 tokens per frame, present in both dev-box caches, M manifest `grids`), then the 416x1024 encode (PI go) |

Because the probe detected at both seeds, WH-NOT-DETECTED would contradict a MEASURED result; the a-W step exists to locate that contradiction before any new lever is spent.

**Country, surface, illumination, period_of_day (each v)**

| outcome | condition | next |
|---|---|---|
| **POOL-REPLACES(v)** | H DETECTS(v) at both seeds (country also BEYOND-RIG) AND H - shipped is not separated on the worse side at either seed | v is reported from the pooled head (one output path for v) |
| **POOL-GAIN(v)** | POOL-REPLACES(v) AND H - shipped separated better at both seeds AND abs(delta) > F_fit | as POOL-REPLACES, and the gain is named with its per-class recall |
| **SHIPPED-KEPT(v)** | otherwise | v stays reported from the shipped `ctx_head` (its STRAT1 DETECTS stands); the pooled sub-head for v is reported beside; the model carries both heads |

The per-variable output map is written by the reader from these rules into `ctx_output_map.json`; nothing is chosen by hand.

**Driving (form (a))**

| outcome | condition | next |
|---|---|---|
| **NO-COST-BY-IDENTITY** | T-ISO (d) byte-equal at both seeds | the head ships with no driving read beyond the identity |
| **ISO-FAIL** | any planner array differs | STOP (instrument stage); nothing ships |

**Beside, with no bar**: FEAT and C0A rows (what the cached features carry with a head-specific projection; whether S1's ctx loss changed the adapter); the probe-vs-H weather row.

## 8. Fallback form (b) (runs only on its trigger in section 7; outline committed here, full card on trigger)

* **Arms** (LoRA pod, one job at a time, the STRAT1 recipe and seeds): S1P_s0, S1P_s1 = S1 with the shipped `ctx_head` replaced by the five-sub-head pooled head (`--ctx-head-form pool`, a NEW trainer flag with its G-DVB entry, default off = `a4357e79`), `w_ctx 0.5`, head lr x10, `--ctx-detach off`; S1P_rep = S1P_s0 flags at training seed 2 (floor). Controls: C0_s0, C0_s1, C0_rep banked (same venue, same code with the flag off; reusable only if the off snapshot test passes).
* **Bars**: detected info as section 6 with K4 on S1P; driving: S1P - S1 and S1P - C0 on the 24 cells and 6 route cells at both seeds, cost or gain = separated and above floor F (C0_rep, as STRAT1). KEEPS-B = no cost at both seeds in S1P - S1 -> S1P's head ships; COSTS-B = a cost at both seeds -> S1P with `--ctx-detach on` (built, default off); a cost at one seed only -> seeds 10 and 11 as the STRAT1 SEEDS-DISAGREE rule.
* **Reachability**: the full trainer trace (argparse -> cfg -> model -> forward -> `T["ctx"]` -> total -> backward), non-zero gradient on the pooled head, exact 0.0 on a control, mutation RED, default named in argv; launch through a `strat1_check.py`-style token.

## 9. Cost

* **Form (a), dev box.**
  * GPU fits (E): one 4-class probe sub-head ran about 9-10 s per epoch on the pod GPU (M); five sub-heads share the frozen adapter projection (about 42 M multiply-adds per window) and add five attention poolings (about 21 M each), about 2.3x the probe per epoch (E from the operation count). H_s0 and H_s1: E selection 0.8 x (4 + 8 + 16) = 22.4 epochs plus finals of at most 4 + 8 + 16 = 28 epochs, at most about 50 epochs each; the nine other fitted arms (RA x2, R1 x2, H_s0_rep, FEAT x2, C0A x2) at most 28 epochs each, FEAT counted 3x (it trains its projection). Upper bound about 465 epoch-equivalents at about 22 s each: **at most about 2.8 h on a pod-class GPU, about 3-6 h on the dev-box GPU** (E; the dev-box GPU model is U here; arms whose E_v coincide cost less). Under the shared lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (below 3,500 MiB; the probe peaked at 467 MiB, M), launched detached by WMI (CreateFlags 0x08000200).
  * Feature load: 4,572 clip files of about 20.9 MB from F: (about 95 GB read once, M file size x count; time E 5-15 min); 36k windows held as float8, about 5.9 GB host RAM (E).
  * CPU: T0 dumps of the H and control arms (about 10 min each, E); T-ISO (d) T1 dumps, 4 x about 10 min (I, OBD README, E); reads under 30 min (E).
  * Build: about half a Sonnet agent-day (E): the module and flag, the fit script, two evaluator flags, the reader wrapper, the checker and about 15 tests.
  * Disk: the head checkpoints are S1 (86,971,541 bytes, M for C0_rep's file) plus about 5.3 MB of head tensors each (E); temporary feature tensors live in RAM or under `F:/TanitAD-scratch/strat1/whead/work/` and are deleted at the end of the run.
  * **Pod time: none. Box-2 time: none.**
* **Form (b), only on its trigger**: 3 pod runs at the STRAT1 card's budget of 0.77 s/step, about 45 min each with gate (E, card section 7), about 2.3 pod-h, plus about 1.5 h of dev-box reads (E) and about one Sonnet agent-day of build (E). The STRAT1 pilot's measured pod rate replaces this estimate (U here). Needs a PI training go (Q1) and a slot between the data-scale blocks. A pilot-projected chain more than 50 % above this estimate stops and goes back to the PI.

## 10. Known risks

* The fit cache may differ in bytes from the pod cache S1 trained on (P-CACHE). The probe showed an encode-source sensitivity of 0.0004 on the eval side (+0.0784 dev-box vs +0.0788 pod-encoded, M); a fit-side difference of that size cannot move a bar of this size, and the count is printed.
* One fit replicate makes F_fit a weak floor; WH-ADAPTER-ADDS needs separation at both seeds as well, so a single lucky fit cannot set the wording.
* Minority weather classes (fog 831, rain 1,647, snow 2,585 train windows, M) are where the label is least checked against the PI (36/40 overall, no per-class PI audit, I). Balanced accuracy and per-class recall are reported beside every row; Q3 is the lever.
* The random adapter can make the trained-vs-random contrast look small because a random 1024 -> 256 projection keeps most linear structure of the features (H). That is the reason WH-INPUT-FEATURES is a wording outcome and not a failure.
* E selection may prefer 16 for some variables (the probe was still rising at 8). The grid is fixed; a still-rising curve at 16 is reported, not extended.

## Log
* 2026-11-16, Architecture agent: drafted before any build, fit or score. Not frozen.
* 2026-11-16 ~22:45Z, MM: FROZEN. Rulings: form (a) covered (Q4 not a blocking question; the PI is told and can stop it); gate token required; fit on the dev-box train cache with its encoder provenance printed; all five variables accepted; base `a4357e79`. Added: P-COMMIT (stream from the clip files when commit is short, byte-equal to the in-RAM path by test), dev-box order and lock, `H - probe` on the dev-box-cache rows only. Text in `MM_RULINGS_WEATHERHEAD.md`.
