<title>REFe review 3 — a fresh source-and-measurement review after the fix round</title>

# REVIEW 3 — REFe against DriveZero, reviewed fresh from source

**Reviewer:** third independent review agent · **Date:** 2026-09-20 · **Window:** 21:22–21:52 +0200
**Method.** Every row was read from SOURCE and, wherever it could be, MEASURED by running code —
against the released DriveZero technical report (text extracted from the PDF itself, not from any
summary), the released DriveRL calculators, the real DINOv3 checkpoints on D:, a real banked
simulation log, and the live scorer bank. The package's prose was never used as evidence that the
code does something. **No instrument written by the package was accepted as proof of the fix it
guards** — every load-bearing verdict below was re-derived with an independently written probe
carrying its own controls.

## ⭐ READ STAMP — and, for the first time in this series, the package DID NOT MOVE

Reviews 1 and 2 both had to pin their findings to md5s because the package was edited five and six
times mid-review. **Every file below was byte-identical at 21:22:48 and at 21:50:03.**

| file | md5 (32) | file | md5 (32) |
|---|---|---|---|
| `refe/model.py` | `61438364f580e1a161894aeefe8e94af` | `refe/diag_rope.py` | `ffbc4663ef82f6e034f39c7ed286b27e` |
| `refe/score_proposals.py` | `7750e814890d08c5bc63c5e12bb094c1` | `refe/diag_architecture.py` | `a26ed0d54d17c7423724a12eb73028a9` |
| `refe/planner.py` | `257b4d6fe6fc89786e12d5972a1e093c` | `refe/scorer_gate.py` | `d301790293f93d1ff006fa22ec07a439` |
| `refe/train.py` | `7affe96ccd25439688061b2354f613e7` | `refe/diag_scorer_components.py` | `bfbabd244a35230779bceb1e16c988b2` |
| `refe/build_scorer_targets.py` | `8a0ca5327ba4ff9cd6d2efbdead9a3b9` | `refe/diag_planner_holds.py` | `0be23b944fb3cec98585a8bfcd8d313f` |
| `refe/validate_model.py` | `d00e33b57ac8e2dc22c98fd73d9b9d56` | `refe/load_dinov3.py` | `44abd8c84e55e5d6612f46f3ecf56df3` |

⚠️ **THE SCORER BANK IS A LIVE, GROWING ARTIFACT.** `D:/Projects/TanitAD/data/refe_scorer_targets_full/`
held **1,033 rows at 21:34:01**, **1,066 parsed at 21:36**, **1,793 at 21:50:03** — one shard
(`scorer_targets.jsonl`), **rank 0 only**, `stride 2`. Every bank number below carries its stamp.
There is **no `scorer_targets_stats.json` yet** (the builder writes it only on completion).

⭐ **A NOTE ON THE PAPER.** Reviews 1 and 2 worked from supplied excerpts. This review reads the
report directly: the PDF's content streams were zlib-decompressed and the text-showing operators
extracted (`scratchpad:paper_text.txt`, 301,223 chars). Every `PUBLISHED` row below is therefore a
**primary** read, including **Table A12**, which reviews 1 and 2 did not have and which contains one
hyper-parameter we are getting wrong.

---

## 1 · Headline verdict

**The fix round worked. Every defect review 2 called WRONG is now genuinely repaired, and I verified
each one with my own instrument rather than theirs.** The rotary encoding is now **bit-identical**
to the released formulation (max |Δ| exactly `0.000e+00` at four geometries) and reproduces the
reference trunk **exactly** on the real ViT-S checkpoint (rel L2 `0.000000`, cos `1.000001`, against
the broken version's 0.760/0.688). The registers really compress (16 scene tokens vs 1,920 visual,
different storage). Progress really spans the route (116 distinct values, varies in 100 % of
frames — it was capped at ~4 % of real progress). DDC really fires (`reverse` 0.9072). The
kinematics derivation is **right in its two main channels**, verified against the log's own recorded
ego states rather than against itself. The parameter budget matches the PI's accepted decision to
the digit: **329,664,066 / 26,584,642 (8.064 %)**, `reg_compress` **12,598,272**.

**But REFe is not yet ready to produce a published number, for four measured reasons, and the first
of them is new — it was found by neither earlier review, it is in the shipped forward pass, and it
is of exactly the class that blocked review 2.**

| | |
|---|---|
| ⛔ **BLOCKING (would change a published number)** | **R3-1** the frozen DINOv3 trunk is fed **un-normalised** `[0,1]` images — the checkpoint's own `pretrained_cfg` declares ImageNet mean/std and **nothing in `refe/` applies it**. MEASURED displacement rel L2 **0.562** / cos **0.841** — *larger than deleting the positional encoding entirely* (0.529 / 0.862) and 2.9× a BGR/RGB swap (0.194) · **R3-2** the scorer bank's collision targets are manufactured by **frozen NPCs** (MEASURED: ego moves 6.340 m while all 127 NPCs move **exactly 0.000000 m**; splicing the log's real futures back in removes up to **16.7 pp** of collisions and adds none) · **R3-3** comfort's entire ranking signal is a **degenerate-prefix / wrong-estimator artefact** (banked: lateral family **0.5000**, identical to the deliberate violator; restricted to the prefixes Comfort scores with its intended estimator: **0.8333**, identical to the teacher) · **R3-4** `Comfort` and `CenterLine` **re-differentiate our 0.2 s values on a 0.1 s grid**, and **4 of the 7** stateful calculator attributes are not carried across prefixes |
| ⚠️ **SERIOUS (would change an operational decision)** | trainable params are **43 % above** the paper's 18.58 M on ONE camera against their four (PI-accepted), but the package's **"fidelity cross-check" now inverts**: the 4-camera reconstruction is **9.693 %** against their published 5.49 %, not the 5.52 % three documents still quote · peak CUDA memory MEASURED **9.58 GB at batch 2** on an **8.0 GiB** card (docs say 3.26/5.36 GB) — the pod provisioning arithmetic rests on the stale figure · **weight decay is 1e-4 against the paper's Table A12 value of 0.01** (100×) |
| ⭐ **CONFIRMED FIXED** | rotary encoding (bit-exact) · register compression · progress/EP · DDC supervision · `--rank` join · WTA L1 over points · LR 2e-4 cosine→0 · decoder 4 layers · epochs→steps · `validate_model.py`'s regression arm · comfort's *inversion* (teacher is now the most comfortable, 0.8454 vs `stopped` 0.5825 — review 2 measured the opposite) |

### ⛔ ESCALATION — three things need a PI decision, not a merge

1. ⛔ **R3-1 must be fixed before any REFe number is produced, and it must be fixed BEFORE the
   backbone ablation is run.** REFe exists to compare DINOv3 against DriveVFM. A frozen trunk fed
   the wrong input distribution is not the trunk being compared. This is the same class as the
   rotary defect — invisible to every tensor-consumption check, measurable only by comparing the
   trunk against itself. One line in two files; the constants are already on disk in the
   checkpoint's own `config.json`.
2. ⛔ **The scorer bank currently being built will have to be rebuilt.** R3-2, R3-3 and R3-4 are all
   *target-generation* defects: they are baked into every row. The build at 21:50 is 1,793 rows of
   rank-0 targets that carry a biased NC, a comfort value produced by the wrong estimator, and a DDC
   window whose width depends on `--stride`. Spending more CPU on this bank before the fixes land
   spends it twice. **This is a scheduling decision, not a code decision.**
3. ⚠️ **The reproduction claim needs its scope restated.** Three documents assert the architecture
   is "dimensionally right" because a 4-camera reconstruction lands within **0.03 percentage
   points** of the published 5.49 %. MEASURED now: **9.693 %** — trainable params are **+75.1 %**
   over 18.58 M, not +4.1 %. The PI accepted the *parameter cost* of compressing at width 1024; the
   PI has not been told that the accepted cost **voids the package's only independent check that
   the architecture is dimensionally faithful**. That check needs either a different formulation or
   a retraction.

---

## 2 · Conformance table

Verdicts: **MATCHES** · **DIFFERS-DELIBERATELY** (with where it is stated) · **DIFFERS-UNINTENDED**
(a defect) · **NOT-IMPLEMENTED** · **UNVERIFIED**. `file:line` is relative to the package root.
Paper quotations are PRIMARY (extracted from `drivezero_report.pdf` this session).

### 2.1 · Architecture (§2.3 and Table A12)

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| trajectory proposals | 64 | 64 (MEASURED forward `(2,64,20,3)`) | `refe/model.py:80` | **MATCHES** |
| planning representation | 256-dimensional | 256 end-to-end | `refe/model.py:85` | **MATCHES** |
| proposal decoder | 4 layers | 4 | `refe/model.py:86` | **MATCHES** |
| scoring decoder depth | *paper silent* | mirrors `dec_depth`, silence documented | `refe/model.py:93`, `:369` | **MATCHES** (a choice, labelled as one) |
| register tokens | 16 per camera | 16 | `refe/model.py:79` | **MATCHES** |
| registers **COMPRESS** into scene tokens | "learnable registers then compress these tokens into a small set of scene tokens" | real cross-attention compression. MEASURED contexts: trajectory `(2,16,256)`, scoring `(2,128,256)`, **different storage pointers** | `refe/model.py:315-327`, `:384-388` | **MATCHES** *(was D2, fixed)* |
| compression width | *paper silent on where* | at backbone width 1024, projected to 256 afterwards | `refe/model.py:342-353` | **DIFFERS-DELIBERATELY — PI DECISION 2026-09-20**, and the implementation matches the decision (MEASURED `reg_compress` = **12,598,272**, the exact figure the PI accepted) |
| trajectory decoder → **scene** tokens | scene tokens | `scene_ctx` | `refe/model.py:391-392` | **MATCHES** |
| scoring decoder → **visual** tokens | visual tokens | `visual_ctx.detach()` | `refe/model.py:397-398` | **MATCHES** |
| scoring decoder "encodes each **candidate trajectory** as a score query" | the trajectory | `s = q.detach()` — the decoder's **latent query**, not the output poses | `refe/model.py:396` | **DIFFERS-UNINTENDED** — R3-9 |
| visual tokens enriched with **3D position embeddings** | 3D PE (their ref [50]) | a free learned per-patch table, randomly initialised, trained. **No intrinsics/extrinsics read anywhere in `refe/`** (grep: 0 hits) | `refe/model.py:340-341`, `:376` | **DIFFERS-UNINTENDED** — carried from review 1's D7, still open |
| ego kinematics + command → **one ego token added to M queries** | one token, added | exactly that | `refe/model.py:389-390` | **MATCHES** |
| the command is "derived from the goal point given to the teacher" | a derived command | the **raw 2 goal points** are embedded | `refe/model.py:355-357`, `:389` | **DIFFERS-UNINTENDED** (partially documented at `:355`) |
| backbone frozen, **rank-32 Q/V LoRA only** | frozen + Q/V LoRA r=32 | MEASURED: **96** trainable backbone tensors, **3,145,728** params, **0** outside `attn.q`/`attn.v` | `refe/model.py:78`, `:190-195` | **MATCHES** |
| positional encoding of the frozen trunk | DINOv3 axial RoPE | MEASURED **bit-identical** to the released formulation at 4 geometries (max\|Δcos\|, max\|Δsin\|, max\|Δapply\| all `0.000e+00`); on the real ViT-S checkpoint **rel L2 0.000000 / cos 1.000001** vs the reference | `refe/model.py:126-168` | **MATCHES** *(was F7-WRONG, fixed)* |
| **image preprocessing of the frozen trunk** | *not in the paper; the checkpoint states it* | **NO normalisation.** `cv2.resize` (bilinear) then `/255.0`. Checkpoint declares `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] interpolation=bicubic` | `refe/train.py:293`, `refe/planner.py:260` vs `D:/Projects/TanitAD/data/backbones/dinov3-vitl16/config.json` | ⛔ **DIFFERS-UNINTENDED — R3-1, BLOCKING** |
| total / trainable | 338.46 M / 18.58 M / 5.49 % *(ViT-L, 4 cameras)* | MEASURED **329,664,066 / 26,584,642 / 8.064 %** *(ViT-L, 1 camera)* | measured by instantiating `refe/model.py` | **DIFFERS-DELIBERATELY** (PI-accepted); ⚠️ but the 4-camera reconstruction is **335,611,458 / 32,532,034 / 9.693 %**, not the 5.52 % three docs quote — see §5 |
| cameras | four (F0, B0, L0, R0) | one (CAM_F0). `n_cameras>1` **still raises** (MEASURED: `size of tensor a (128) must match b (256)`) | `refe/model.py:75`, `:376` | **DIFFERS-DELIBERATELY** (PI); the raise is a latent trap, R3-12 |
| backbone | DriveVFM ViT-L | DINOv3 ViT-L | `refe/model.py:67` | **DIFFERS-DELIBERATELY** (PI) |

### 2.2 · Training (Table A12, read PRIMARY this session)

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| prediction horizon | 20 steps at 5 Hz | 20 @ stride 2 × 0.1 s = 4.0 s | `refe/model.py:81`, `refe/build_targets.py:53` | **MATCHES** |
| optimizer | AdamW | AdamW | `refe/train.py:378` | **MATCHES** |
| initial learning rate | 2 × 10⁻⁴ | 2e-4 | `refe/train.py:517` | **MATCHES** |
| LR schedule | "cosine annealing to zero (no warmup)" | `CosineAnnealingLR(T_max=steps, eta_min=0.0)`, stepped per optimiser step. MEASURED reaching **0.000e+00** | `refe/train.py:385`, `:452-453` | **MATCHES** |
| epochs | 25 | `--epochs N` derives steps **before** the optimiser is built | `refe/train.py:366-371` | **MATCHES** (default is `--steps 300`; 25 is reachable, not default) |
| batch size | 256 | default **8**, deferred to a pod measurement | `refe/train.py:516` | **DIFFERS-DELIBERATELY** (deferred, stated) |
| **weight decay** | **0.01** *(Table A12)* | **1e-4** | `refe/train.py:519` | ⛔ **DIFFERS-UNINTENDED — R3-8.** ⚠️ Review 1 recorded "weight decay: not stated — not a conformance item". **That is refuted:** the paper states it in Table A12 |
| gradient clipping | global norm 1.0 | `clip_grad_norm_(..., 1.0)` | `refe/train.py:450` | **MATCHES** |
| training data | 100K navtrain + 237K SimScale | nuPlan mini, 2,182 tuples over 10 logs × 2 ranks | `REFE_PLAN.md:20` | **DIFFERS-DELIBERATELY** (a stated staging step) |

### 2.3 · Losses and supervision

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| teacher rollout under log-replay agents | "rolled out with all background actors following the driving log" | non-reactive log replay (`val14_nr` → `box_observation` → `TracksObservation`) | `code/run_mini_teacher.sh:18`, `:63` | **MATCHES** |
| `L_traj = min_m dist(τ̂_m, τ_T)` | winner-takes-all | WTA over M | `refe/model.py:419-421` | **MATCHES** |
| `dist` = "the **L1 distance averaged over points**" | over points | `(traj[...,:2]-tgt[...,:2]).abs().sum(-1).mean(-1)` — L1 over the position channels, mean over points. Yaw is a **separate** weighted term on the selected proposal | `refe/model.py:419`, `:423-427` | **MATCHES** on the selection metric; the `+ 0.1·yaw` term is an **undocumented addition** the paper does not describe — R3-11 |
| `L_score = (1/M) Σ_m Σ_k BCE(σ(ŝ_mk), s_mk)` — **normalised over ALL M** | all M proposals | only the ≤11 proposals nearest a banked candidate are supervised; **typically ≥53 of 64 receive no score target** and are unconstrained at `argmax` | `refe/train.py:415-419` | **DIFFERS-DELIBERATELY** in kind (the precompute approximation is stated at `train.py:100-108`) but the **unsupervised-proposal consequence is not stated** — R3-10 |
| candidates **detached** before the scoring branch | detach | `q.detach()` **and** `visual_ctx.detach()` | `refe/model.py:396`, `:398` | **MATCHES + an undocumented extra** |
| the six components | collision avoidance · drivable-area compliance · progress · time to collision · comfort · driving-direction compliance | exactly those six, in that order | `refe/train.py:127` | **MATCHES** *(was mis-mapped; fixed)* |
| aggregation "according to the benchmark scoring rule"; inference picks the highest predicted aggregate | PDM rule | `sigmoid` → `NC·DAC·DDC·TLC(=1) × (5·EP + 5·TTC + 4·C)/14`, `argmax` | `refe/planner.py:226-233`, `:242` | **MATCHES in structure**; TLC is a constant 1.0 and one comfort stands for LK+HC. Correctly labelled "NOT an EPDMS" at `:202-204`. ⚠️ the comment at `:194-201` says **denominator 12** while `:205` is `(5,5,4)` summing to **14** — R3-13 |
| goal augmentation 3-way consistency | "the same augmented route ... keeping the navigation command, teacher trajectory, and proposal scores mutually consistent" | `--rank` written into every row; `ScorerBank` keys `(log, token, step, rank)`; legacy rank-less rows counted | `refe/build_scorer_targets.py:113`, `:278`; `refe/train.py:173` | **MATCHES in mechanism** *(was D1, fixed)*. ⛔ **But there is no rank-1 bank**: at 21:50 the bank is rank 0 only, so the fix converts silent *mislabelling* into silent *halved coverage* — R3-7 |

### 2.4 · Scorer-target generation (no paper counterpart — this is our rig)

| item | intent | our value | file:line | VERDICT |
|---|---|---|---|---|
| candidate sample spacing | 0.2 s (5 Hz) | `TRAJ_DT = 0.2` | `refe/score_proposals.py:57` | **MATCHES** — and verified against the log, not against itself (§3, R3-5) |
| history rate for the derivative seed | one feature-builder row | `HISTORY_STEP_ROWS = 1`. MEASURED from the dataclass: `history_sample_interval = 0.2`, `history_steps = 5` ⇒ one row **is** one `TRAJ_DT` | `refe/score_proposals.py:66` vs `DriveRLNuPlanFeatureBuilderConfig` | **MATCHES** |
| velocity / longitudinal acceleration | derived from the injected poses | MEASURED vs the log's own ego states, n=400: `v` MAE **0.1726 m/s** on a 7.30 m/s mean (corr **0.999**); `a_long` MAE **0.0963 m/s²** (corr **0.987**) | `refe/score_proposals.py:212-232` | **MATCHES** |
| **yaw rate** | derived from the injected yaw | ⛔ `dyaw[:, 0]` is **forced to 0** while acceleration IS seeded from history. MEASURED at the seam: derived **0.0000** for every frame against a logged mean **−0.0357 rad/s**, max error **0.4584**. Excluding the seam, corr rises **0.925 → 0.987** and max error falls **4.9×** | `refe/score_proposals.py:243-246` | ⛔ **DIFFERS-UNINTENDED — R3-4a** |
| lateral acceleration | derived | computed at `:238` and **never written**; `Comfort` hard-zeros it anyway (`comfort.py:397`) | `refe/score_proposals.py:238` | **DEAD** (carried from review 2) |
| the consumer's time base | one time axis | ⛔ `Comfort.time_interval = 0.1` and `CenterLine._frame_time_interval = 0.1` (MEASURED), hardcoded at `scorer_gate.py:105`, against our 0.2 s grid | `refe/scorer_gate.py:105` | ⛔ **DIFFERS-UNINTENDED — R3-4b** |
| prefix aggregation | events max, cumulative last, rewards min | implemented as described | `refe/score_proposals.py:522-538` | **MATCHES in form**; three leaves get the wrong rule — R3-6 |
| `MIN_PREFIX` | "the shortest prefix with a well-defined jerk" | 3. ⛔ The real threshold is Comfort's own estimator switch at history length 15, i.e. **k ≥ 10** with `init_steps = 0` | `refe/score_proposals.py:62` | ⛔ **DIFFERS-UNINTENDED — R3-3** |
| stateful-calculator continuity | reproduce the engine loop | **3 of 7** attributes carried | `refe/score_proposals.py:414-416` | ⛔ **DIFFERS-UNINTENDED — R3-4c** |
| cross-candidate isolation | none | ✅ MEASURED: rescoring the teacher after a foreign candidate changes **0 of 30** leaves; `sd` is never mutated | `refe/score_proposals.py:515` | **MATCHES** |
| background actors while scoring | the paper's protocol is log replay | ⛔ every NPC is **held at its last pose**. MEASURED: ego 6.340 m, all 127 NPCs **0.000000 m** | `refe/score_proposals.py:143` | ⛔ **DIFFERS-UNINTENDED — R3-2, BLOCKING** |

---

## 3 · Defects, ranked by whether they would change a published number

### ⛔ R3-1 — the frozen trunk is fed un-normalised images. NEW; neither earlier review looked.

**Evidence class: MEASURED (ours).** Probe `scratchpad:r3_probe.py`, section C, on the **real
DINOv3 ViT-S checkpoint** (186 tensors loaded, 0 unconsumed, 0 missing — the probe refuses to
measure on a partial load).

`refe/train.py:293` and `refe/planner.py:260` are identical:
```
im = cv2.resize(im, (c.img_w, c.img_h))[:, :, ::-1].astype(np.float32) / 255.0
```
A grep over all of `refe/*.py` for `0.485|0.456|0.406|0.229|0.224|0.225|normalize|Normalize|IMAGENET`
returns **zero** hits. The checkpoint's own `config.json`, sitting in the directory
`load_dinov3.py` reads from, declares `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`,
`interpolation="bicubic"`.

Same trunk, same image, same weights, patch tokens compared:

| arm | rel L2 vs correctly-normalised | cos |
|---|---|---|
| **SELF** *(control, must be 0)* | **0.00000** | 1.00000 |
| **JITTER** *(control, must be small)* | **0.00014** | 1.00000 |
| BGR/RGB swap *(a known-real bad input, for scale)* | 0.19360 | 0.98137 |
| **no positional encoding at all** *(for scale)* | 0.52927 | 0.86244 |
| ⛔ **REFe as shipped (un-normalised)** | **0.56200** | **0.84077** |

**Why this is the top defect.** Review 2 blocked the whole package because the rotary encoding
displaced the trunk by 0.760/0.688 — *worse than no encoding at all*. This defect displaces it by
**0.562/0.841**, which is **also worse than no encoding at all** (0.529), and it is still shipping.
The input range is `[0.000, 1.000]` where the pretrained distribution is `[−2.117, +2.640]`: a shift
of ~0.45 and a scale of ~4.45×, on a **frozen** trunk whose only degree of freedom is a rank-32
Q/V LoRA. REFe's entire purpose is to compare DINOv3 against DriveVFM; a backbone fed the wrong
input distribution is not the backbone being compared, and every PDMS number would be attributable
to the preprocessing rather than to the model.

⚠️ **It is at least self-consistent**: training and inference are both un-normalised, so this is a
wrong operating point, not a train/inference skew. That is the only mitigating fact.

**Minimal fix (described, NOT applied).** In both `refe/train.py:293` and `refe/planner.py:260`,
after `/255.0`, subtract the mean and divide by the std **read from the checkpoint's `config.json`**
rather than hardcoded — `load_dinov3.py` already opens that directory, so the constants should
travel with the weights and a backbone swap cannot silently keep the wrong ones. Change
`cv2.resize` to `interpolation=cv2.INTER_CUBIC` to match `pretrained_cfg`. **Add the arm to
`diag_rope.py`'s table**: it is the same instrument and the same question.

---

### ⛔ R3-2 — the scorer bank's collision targets are manufactured by frozen NPCs

**Evidence class: MEASURED (ours).** Probes `scratchpad:r3_scorer.py` §3 and `r3_scorer2.py` §A.

`inject_proposal` appends the candidate's poses and **holds every other agent at its last recorded
pose** (`refe/score_proposals.py:143`, `add = pos[:, :, -1:, :].repeat(1, 1, T, 1)`). MEASURED on a
real frame: over the 20-step horizon the ego travels **6.340 m** while the NPC displacement is
**max 0.000000 m, mean 0.000000 m over 127 agents**.

⭐ **The futures are already in hand and unused.** `LogSession.scenario_data` returns
`log_sd` from `build_log_scenario_data(sd, sc, step, HORIZON)`, and MEASURED its
`agent_positions_all` is **`(1, 128, 25, 2)`** against `sd`'s `(1, 128, 5, 2)` — it carries the
whole horizon.

**The discriminating experiment, with a control that could have refuted the claim.** 12 frames × 11
candidates, endpoint scoring, everything identical except the NPC rows: held vs spliced from
`log_sd`.

| candidate | HELD NPCs | LOG-REPLAY NPCs | delta |
|---|---|---|---|
| `lon x0.5` | 50.0 % | 33.3 % | **+16.7 pp** |
| `lat+2` | 25.0 % | 8.3 % | **+16.7 pp** |
| `lat+4` | 25.0 % | 8.3 % | **+16.7 pp** |
| `reverse` | 33.3 % | 25.0 % | +8.3 pp |
| `lat-4` | 25.0 % | 16.7 % | +8.3 pp |
| `teacher`, `lon x1.5`, `jerky`, `lat-2`, `stopped`, `over-curb` | — | — | 0.0 pp |

**Holding the NPCs adds collisions for 5 of 11 candidates and removes none.** The bias is not
uniform — it lands on the lateral and slow candidates — so it **distorts the ranking**, not merely
the level. NC is a **multiplier** in the selection rule, so a spurious NC=0 makes a candidate
unselectable no matter how good the rest are.

⚠️ **Honest limits.** (a) n = 12 frames on one run; (b) my splice arm is a *discriminating control*,
not a proposed implementation — it replaces NPC positions, orientations and velocities and does not
re-derive their polygon history; (c) the full rollout would amplify the effect, since the `max`
aggregation gives every prefix another chance to fire. What is established is the **direction and
the selectivity**, both one-sided.

⭐ **And the paper settles the protocol question.** DriveZero's own teacher is "rolled out with **all
background actors following the driving log**". The bank's docstring says it "scores the proposal
against the recorded scene" (`score_proposals.py:24-26`) — but the recorded scene has the actors
*moving*. Freezing them is not a smaller approximation than log replay; it is a different and
harsher one, and it is undocumented.

**Minimal fix.** In `inject_proposal`, when a `log_sd` carrying ≥ `T0+T` steps is available, fill the
appended NPC rows from it instead of repeating the last pose; keep the hold as the explicit fallback
and **record which was used in the row**, so a bank built either way is identifiable. State the
protocol in `build_scorer_targets.py`'s docstring beside the "does NOT simulate other agents
reacting" sentence, which is true of both variants and therefore does not distinguish them.

---

### ⛔ R3-3 — comfort's ranking signal is a degenerate-prefix / wrong-estimator artefact

**Evidence class: MEASURED (ours).** Probes `scratchpad:r3_scorer.py` §1 and `r3_scorer2.py` §B.

`Comfort.data_preprocessing` chooses between **two different estimators** at
`comfort.py:430`: `if a_long_history.shape[-1] >= 15` → Savitzky-Golay jerk; else a crude 2-point
difference. MEASURED: `init_steps = 0`, so the history slice is the whole tensor and its length is
`5 + k`. ⇒ **prefixes k = 3…9 use the crude estimator and k ≥ 10 use savgol.** The `min` over
prefixes mixes the two.

Per-prefix comfort on one real frame (stride 2, exactly as the live bank was built):

```
candidate    k03 k05 k07 k09 | k11 k13 k15 k17 k19 k20   banked min
teacher      1.0 1.0 1.0 1.0 | 1.0 1.0 1.0 1.0 1.0 1.0      1.00
lat-4        0.5 1.0 1.0 1.0 | 1.0 1.0 1.0 1.0 1.0 1.0      0.50   <-- k=3 alone
lat-2        0.5 1.0 1.0 1.0 | 1.0 1.0 1.0 1.0 1.0 1.0      0.50   <-- k=3 alone
lat+2        0.5 1.0 1.0 1.0 | 1.0 1.0 1.0 1.0 1.0 1.0      0.50   <-- k=3 alone
lat+4        0.5 1.0 1.0 1.0 | 1.0 1.0 1.0 1.0 1.0 1.0      0.50   <-- k=3 alone
jerky        0.5 1.0 1.0 1.0 | 1.0 1.0 1.0 0.5 1.0 1.0      0.50
over-curb    1.0 1.0 1.0 1.0 | 0.5 0.5 0.5 0.5 0.5 0.5      0.50
             ^^^^^^^^^^^^^^^ crude 2-point jerk at dt=0.1
```

Aggregated over 9 frames × 11 candidates:

| candidate | **min over all prefixes (what the bank stores)** | **min over k ≥ 11 (savgol only)** | k = 3 alone |
|---|---|---|---|
| teacher | 0.8333 | **0.8333** | 0.8889 |
| lat-4 / lat-2 / lat+2 / lat+4 | **0.5000** | **0.8333** | 0.5000 |
| `jerky` *(the deliberate violator)* | **0.5000** | **0.6111** | 0.5000 |
| over-curb | 0.6111 | 0.6111 | 1.0000 |
| stopped | 0.7222 | 0.7222 | 1.0000 |

⛔ **Banked, the four lateral candidates are indistinguishable from the deliberate comfort
violator. Restricted to the prefixes Comfort scores with its intended estimator, they are
indistinguishable from the teacher — and `jerky` separates cleanly.** The component's apparent
discrimination (it varies in 85.6 % of frames on the live bank) is produced almost entirely by the
degenerate short prefixes.

⭐ **`MIN_PREFIX = 3` did not fix review 2's boundary artefact; it moved it.** Review 2 measured
"at k=1 every candidate reads 0.5". The raise to 3 was reasoned from the *derivative* ("two
differences for acceleration, three poses for jerk") — but the threshold that matters belongs to the
**consumer**, and it is 15 history rows, i.e. **k ≥ 10**. This is the file's own lesson — *find what
the consumer reads before deciding what to write* — extended one step: **find where the consumer
switches algorithm.**

⚠️ **The candidate set contributes its own half of this.** `xy[:, 1] += torch.linspace(0, d, T)`
puts zero offset at index 0 and a constant lateral velocity from index 1, so the candidate has a
**one-step lateral acceleration transient** of `d / ((T−1)·dt²)` = **2.63 m/s² for d=2** and
**5.26 m/s² for d=4** at the start of the horizon — an artefact of how the perturbation is built,
not a property worth supervising. That is what the crude branch is reading.

**Minimal fix.** (a) Derive `MIN_PREFIX` from the consumer rather than choosing it:
`MIN_PREFIX = 15 - history_steps + init_steps` (= 10 here), and **assert** it rather than
commenting it, so a config change moves it. (b) Build the lateral family with a C¹ profile (a
smoothstep or minimum-jerk lateral offset) so the perturbation does not inject a step in lateral
velocity. (c) Print the estimator branch per prefix in `diag_scorer_components.py` so a future
reader sees the switch.

---

### ⛔ R3-4 — the rig runs on two time bases, and carries 3 of 7 stateful attributes

**Evidence class: MEASURED (ours) + PRIMARY source.**

**(a) The yaw-rate seed is not seeded.** `refe/score_proposals.py:243-246` writes
`dyaw = zeros_like(yaw); dyaw[:, 1:] = yaw[:,1:] - yaw[:,:-1]`, so `yaw_rate[0] = 0` — while the
acceleration path goes to considerable length to seed itself from history (`:222-229`, with ~30
lines of comment explaining why a zero seed is a defect). MEASURED over 20 frames against the log's
own `angular_velocity`:

| window | derived mean | logged mean | MAE | corr | max\|err\| |
|---|---|---|---|---|---|
| all 20 candidate steps | −0.0114 | −0.0119 | 0.0069 | 0.925 | 0.4584 |
| **interior (steps 2–20)** | −0.0120 | −0.0106 | **0.0051** | **0.987** | **0.0927** |
| **the seam (step 1 only)** | **0.0000** | −0.0357 | 0.0418 | *nan — derived has zero variance* | **0.4584** |

The entire degradation of the yaw channel is one forced zero. **Consequence:** `Comfort` computes
`yaw_accel = Δyaw_rate / time_interval` (`comfort.py:419-421`) and then corrects `a_long` by
`rear_axle_to_center × yaw_accel` (`:427-429`). At the measured worst case that is
`0.4584 / 0.1 × 1.461` = **6.70 m/s² of pure artefact**, against Comfort's 2.4 m/s² gate — enough to
manufacture a comfort violation by itself. On the straight frames I could score it is small
(0.055 m/s²); it **scales with the logged yaw rate**, so it is a turning-frame defect.
*Fix: seed `dyaw[:, 0]` from the last history yaw exactly as `v_prev` is seeded.*

**(b) The consumers differentiate on 0.1 s; we write on 0.2 s.** MEASURED: `frame_time_interval` is
**hardcoded 0.1** at `refe/scorer_gate.py:105`, so `Comfort.time_interval = 0.1` and
`CenterLine._frame_time_interval = 0.1`. Every second derivative Comfort computes from our values is
**2× too large**. And `CenterLine`'s direction window is `round(1.0 / 0.1) = 10` slots — one slot per
**prefix**, i.e. **2.0 s at stride 1 and 4.0 s at stride 2**, against an intended 1.0 s. ⛔ **The DDC
target therefore changes meaning with `--stride`, a flag documented purely as a cost knob**, so
shards built at different strides are not comparable. *Fix: pass `frame_time_interval=TRAJ_DT` when
building the engine config — one line, and it makes one time axis serve everything. Then re-check
the DDC thresholds, which were tuned for a 1.0 s window.*

⚠️ Also: `_derive_ego_kinematics` writes `agent_jerk_long_all`, but in the savgol branch Comfort
**never reads it** — it re-derives jerk from the acceleration history (`comfort.py:437-443`) and
reads `agent_jerk_long_all` only in a fallback that needs `< 2` history rows (`:456`). The carefully
derived longitudinal jerk is dead on the main path.

**(c) Four of seven stateful attributes are dropped.** Enumerated from the released source
(`grep -rn "scenario_data\._[a-z_]* ="`):

| attribute written on `scenario_data` | calculator | carried by `DIRECTION_STATE`? | what is lost |
|---|---|---|---|
| `_direction_progress_last` / `_direction_baseline_last` / `_direction_progress_buffer` | CenterLine | ✅ yes | — |
| **`_comfort_triggered`** | Comfort | ⛔ **no** | the repeat-offender latch: `comfort_score × 0.75` on any later violation (`comfort.py:171-186`) |
| **`_nuplan_ttc_triggered`** | NuPlanTTC | ⛔ **no** | `ttc_reward × 0.5` on a repeat trigger (`nuplan_ttc.py:167-173`) |
| **`_nuplan_ttc_collided_matrix`** | NuPlanTTC | ⛔ **no** | the accumulated excluded-collision set (`nuplan_ttc.py:147-151`) |
| **`_cross_lane_deadband_counter`** | OffRoad | ⛔ **no** | the cross-lane deadband (`off_road.py:189-206`) |

**Two of the four belong to components REFe actually consumes** (comfort and TTC, the latter
weighted 5/14 in the selection rule). `min` over prefixes does **not** substitute for the latches:
the engine would report `{1.0, 0.5, 0.375}` where our rig reports `{1.0, 0.5, 0.5}`. ⇒ **the banked
comfort and TTC targets are systematically optimistic for any candidate that violates more than
once.** *Fix: extend `DIRECTION_STATE` to all seven and rename it (it is no longer about direction);
the per-rollout reset in `score_proposal_rollout` already gives the right isolation — MEASURED, see
below.*

✅ **The isolation itself is correct, and I could not break it.** `carry` is created fresh per
rollout, so it cannot leak between candidates. MEASURED with a direct test — score the teacher,
score a foreign candidate, rescore the teacher: **0 of 30 leaves differ**, and `sd` itself is never
mutated (`_direction_progress_last` is absent from it afterwards). The within-rollout threading is
also semantically right: at prefix *k* the ego sits at `traj[k−1]`, so prefix *k+1* computes exactly
the engine's one-step delta. **The defect is incompleteness, not contamination.**

---

### ⛔ R3-5 — *(not a defect — reported because the brief asked it to be attacked)* the kinematics ARE consistent with the log

The brief's first item was *"a plausible-looking speed is not evidence"*. So I did not check the
derivation against itself; I checked it against the simulation log's own recorded ego states at the
same timestamps, which is an independent reference because the teacher candidate's poses **are** the
log's future. `scratchpad:r3_kin.py`, 20 frames × 20 steps = 400 samples:

| channel | derived mean | logged mean | MAE | corr | max\|err\| |
|---|---|---|---|---|---|
| speed | 7.1575 m/s | 7.2951 m/s | **0.1726** (2.4 %) | **0.999** | 0.4468 |
| `a_long` | 0.9335 m/s² | 0.8973 m/s² | **0.0963** | **0.987** | 1.9838 |
| yaw rate | −0.0114 rad/s | −0.0119 rad/s | 0.0069 | 0.925 | 0.4584 *(see R3-4a)* |

**Mutation arm — the check can go red.** Forcing `TRAJ_DT = 0.1` (the defect review 2 confirmed
fixed) makes derived speed **8.0290** against a logged **4.1329** on the same frames — a ratio of
**1.94**, i.e. the expected 2×, MAE **3.90 m/s**, max error **9.15 m/s**. A wrong `dt` is a ~94 %
miss on this reference, so agreement at 2.4 % is meaningful.

**Frame, units, seed, history rate — each checked from the owning object, not from prose:**
* **frame** — ego-centric; the ego's last recorded position reads `[0,0]` and the candidate is
  written into the same array, so no conversion is needed. ✓
* **units** — metres and seconds throughout; `TRAJ_DT` in seconds; yaw in radians, wrapped before
  differencing (`:245`). ✓
* **history rate** — `HISTORY_STEP_ROWS = 1` is **correct**, verified by reading
  `DriveRLNuPlanFeatureBuilderConfig`: `history_sample_interval = 0.2`, `history_steps = 5`. ✓
* **seed** — correct for velocity and acceleration, **absent for yaw rate** (R3-4a). ⛔

---

### ⛔ R3-6 — three leaves get the wrong aggregation rule

**Evidence class: MEASURED (ours) + source.** The brief asked me to name any leaf whose
suffix-dispatch is wrong. I enumerated every leaf the six calculators write
(`off_road.py:238-249`, `center_line.py:264-281`, `comfort.py:169`, `goal_reaching.py:176`,
`nuplan_ttc.py:177-189`, `collision/nuplan_collision.py:48-51`) and checked each against the rule at
`score_proposals.py:522-538`. **Most are right.** Three are not:

| leaf | what it holds | rule applied | why it is wrong | consumed? |
|---|---|---|---|---|
| **`off_road.OffRoad.info`** | a **nominal category** 0 none / 1 off-road / 2 solid-line / 3 wrong-way / 4 severe | `max` | `max` over a nominal enum is undefined. Within one prefix 3/4 is written **only** where the category is not 1 (`center_line.py:202`), so a candidate that is off-road at prefix *a* and wrong-way at prefix *b* aggregates to 3 or 4 — and `ScorerBank.components` reads `cat == 1`, so **DAC reads CLEAN**. MEASURED on the live bank at 21:36: categories `{0: 699, 1: 267, 3: 10, 4: 90}` — **100 of 1,066 rows (9.4 %) are at 3/4 and therefore DAC-clean by construction**. ⚠️ Before the carry fix only `{0,1}` occurred, so **the carry fix is what made this reachable** | ⛔ **YES — DAC, a multiplier** |
| **`collision.NuPlanCollision.reward`** | `collision_info.float() × weight`; MEASURED `collision_reward_weight = −1.0`, so **more negative is worse** | `min` | `min` happens to take the worst here because the weight is negative — but the rule is applied blind to sign, and a positive weight would invert it silently | no (NC reads `.info`) |
| **`ttc.NuPlanTTC.info`** | `stack((ttc_info, TH, TH, TH, TH))`; `_ego_scalar` reduces a vector to its **norm** | `max` | the banked value is `sqrt(ttc² + 4·TH²)` — MEASURED **6.7082**, which is `sqrt(3² + 4·3²)`, not a TTC. A reader would take it for seconds. And `max` takes the **safest** prefix, not the worst | no (TTC reads `.ttc_reward`) |
| `goal_reaching.GoalReaching.next_goal_stage` | an ordinal stage index | `min` | takes the earliest stage reached | no |

⚠️ **Honest limit on the DAC finding.** The *mechanism* is confirmed from source and the *exposure*
is measured (9.4 % of rows). I scored one frame per prefix looking for an actual loss and **did not
observe one**: on that frame `reverse` goes `0→3→4` (never 1) and `over-curb` goes `1→0` (max
correctly 1). ⇒ **MECHANISM CONFIRMED, LOSS NOT OBSERVED AT n = 1.** The fix is two lines and
worth doing regardless, because it also affects the per-frame control: `build_scorer_targets.py:266`
gates on `by["over-curb"]["off_road.OffRoad.info"] == 1.0`, an exact equality on a category that is
now reachable at 3 and 4 — so a curb candidate that also trips the direction detector **aborts its
frame silently**.

**Minimal fix.** Derive the category indicators **per prefix**, inside `score_proposal`, exactly as
`ddc.violation` already is (`score_proposals.py:477-479`): add
`out["dac.violation"] = 1.0 if cat == 1 else 0.0`, have `ScorerBank.components` read that leaf, and
have the `ok_curb` control read it too. Aggregate `off_road.OffRoad.info` only for the record.

---

### ⚠️ R3-7 — the rank join is closed, but there is no rank-1 bank

`--rank` is written into every row (MEASURED: 1,066/1,066 rows carry `rank`, all `0`), the key is the
4-tuple, and legacy rows are counted. ✓ **But at 21:50 the bank is a single rank-0 shard.** A
training run started now would key every rank-1 tuple to a rank-1 bank entry that does not exist,
and `TargetBank` would return `None` — silently halving scorer coverage. The coverage print at
`train.py:465-468` makes it *visible*, and nothing refuses. ⚠️ Also still unfixed from review 2:
`scorer_targets_stats.json` is **not** rank-suffixed (`build_scorer_targets.py:318`) while the data
file is — severity is now LOW, because the rank lives in every row and the stats file is no longer
load-bearing.

### ⚠️ R3-8 — weight decay is 1e-4 against the paper's 0.01

PRIMARY, Table A12: *"Weight decay 0.01"*. `refe/train.py:519` defaults to `1e-4` — **100× smaller**.
Every other Table A12 training row matches (AdamW, 2e-4, cosine-to-zero-no-warmup, 25 epochs,
batch 256, gradient clipping global norm 1.0). ⚠️ **Review 1 recorded "weight decay: not stated —
not a conformance item"; that is refuted by the table.** *Fix: change the default to 0.01 and cite
Table A12 in the help string, as `--lr` already does.*

### ⚠️ R3-9 — the score query is the latent, not the trajectory

PRIMARY: *"A separate scoring decoder then encodes each candidate trajectory as a score query."*
`refe/model.py:396` uses `s = q.detach()` — the decoder's pre-head latent. The scorer therefore never
sees the poses it is scoring, only the state that produced them. It is in the spirit of the next
sentence (*"Candidate trajectories are detached before entering the scoring branch"*), and it is
carried from review 1's D9, but it remains a functional difference and it is undocumented.
*Fix, if taken: encode `traj.detach()` through a small MLP into the score query.*

### ⚠️ R3-10 — ~53 of 64 proposals receive no score target

PRIMARY: `L_score = (1/M) Σ_m Σ_k BCE(...)` — normalised over **all M**. `train.py:415-417` gathers
only the proposals nearest the ≤11 banked candidates. The remaining proposals' six logits are
unconstrained, and `planner.aggregate(...).argmax()` runs over **all 64** at inference. ⇒ an
unsupervised proposal can win. The precompute approximation is honestly stated
(`train.py:100-108`); **this consequence of it is not.** *Fix: state it, and consider a weak prior
(e.g. BCE toward the frame's mean target) on the unassigned proposals, or restrict the argmax to
proposals within an assignment radius.*

### ⚠️ R3-11 — the yaw term and `yaw_w = 0.1` are still unmotivated

PRIMARY: `dist` is *"the L1 distance averaged over points"* over `(x, y, yaw)` poses. REFe splits it:
selection on position only (correct, with a measured counterexample in the docstring), plus a
separate `0.1 × wrapped-yaw` term. The split is defensible and documented; the **constant is not** —
grep over `refe/*.py` and the package's `.md` files returns exactly two hits, the definition and the
use. No CLI flag, no ablation. Carried unchanged from review 2.

### ⚠️ R3-12 — `n_cameras > 1` raises, and `apply_rope` silently no-ops

MEASURED: `n_cameras=2` and `4` both raise `RuntimeError: The size of tensor a (128) must match the
size of tensor b (256)` in `REFe.forward` — `self.pos3d` is sized from `cfg`, the backbone output is
not tiled. Pre-existing; the PI decision is one camera; but the 4-camera **parameter** count is
quoted in three documents as a fidelity check, and it cannot be produced by a forward pass.
Separately, `apply_rope` returns its input **unchanged** when the token count is short
(`model.py:174-175`) — MEASURED `True`. It cannot fire in the trunk today, but it is a silent path
to the exact "no positional encoding" arm that measures 0.529. *Fix: raise instead of returning.*

### ⚠️ R3-13 — `planner.py`'s own comment contradicts its own constant

`refe/planner.py:194-195` and `:200-201` state *"weighted EP(5) · TTC(5) · comfort(2), denominator
12"* and *"which is why the denominator is 12 and not 14"*. `:205` is `PDM_W = (5.0, 5.0, 4.0)` and
`:228-230` divides by `w.sum()` = **14**. The code is right (14 keeps the aggregate on the
benchmark's scale); the comment four lines above it is the superseded version. A paper draft written
from that comment would state the wrong rule.

---

## 4 · Instruments audit — which controls can go red

Every instrument was run (raw output banked, see the manifest) and every arm read from source. The
question asked of each arm was review 2's: **could this control actually go red?**

| instrument | run result | verdict on its controls |
|---|---|---|
| `diag_architecture.py` | `ARCHITECTURE_CONFORMS` 9/9, 2.2 s | **D2 arms: CAN GO RED** — they read live forward hooks and compare `data_ptr`; they caught the real defect. **D7 arms: STILL CANNOT GO RED**, unchanged from review 2 — `backbone.pos` absent / rotation is non-zero / rotation preserves the norm are all true of *any* rotary encoding including a random one, and the header still asks *"is the pretrained position encoding back?"*. **D6's "CONTROL" is arithmetic on the test's own literals** (0.60/3 vs 0.30/3) and would pass with `model.py` deleted |
| `diag_rope.py` | prints; **exit 0 always** | ⛔ **CANNOT GO RED — it has no assertions and no verdict token** (`main()` returns `None`). A regression to the old rope would print `0.760` and still exit 0. ⛔ **And its "REFe" angle-span row is HARDCODED `torch.arange(gh)` / `arange(gw)` at lines 123–124 — it never calls `build_axial_rope`.** It therefore prints `REFe y [+0.000,+7.000] x [+0.000,+15.000]` while the live code's span is `y [−5.498,+5.498] x [−5.890,+5.890]`, identical to the reference row printed above it. **The instrument banked to prove the fix displays the defect as though it were still present** — independently confirmed twice this session. ⚠️ Its `dinov3_rope` is also the author's own retyping of the same seven lines as `model.py`; agreement measures transcription, not conformance. The cited primary (`facebookresearch/dinov3`) is neither imported nor banked |
| `validate_model.py` | `VALIDATE_OK`, 67 s | **The regression arm CAN GO RED** and was honestly rebuilt against the current forward ✓. ⚠️ But it is a **hand-mirror** of `model.forward`, not a call to it, and nothing asserts the mirror still matches — the same drift that broke it before, one level out. ⛔ **Three things print and never reach the verdict:** the published-parameter comparison (8.06 % vs 5.49 %, nothing flags it), `traj_head grad present`, and **peak CUDA memory — MEASURED 9.58 GB at batch 2 on an 8.0 GiB card**. Docstring check 5 ("WTA routes gradient to exactly one proposal") is not implemented |
| `scorer_gate.py --log …` | `SCORER_DISCRIMINATES` | the teacher-clean and over-curb-caught arms **CAN GO RED** ✓. ⚠️ "rollout separates more than endpoint" asserts `>=`, so it passes when the rollout separates *nothing* more |
| `scorer_gate.py --no-enrich` *(the documented regression arm)* | `SCORER_DISCRIMINATES`, exit 0 | ⛔ **CANNOT GO RED — MEASURED byte-identical output except one status line.** `enrich_lane_graph` writes only `lanes_centers_groups/_ids/_next_groups`, which none of the three checks reads. ⭐ **This also refutes `score_proposals.py:98-103`**, which claims an un-enriched graph leaves the calculators inert for every candidate: with the graph absent, `OffRoad.info` still reads 1.0 on the over-curb rollout and every `CenterLine` value is unchanged. `diag_enrich_cost.py`'s own docstring already calls the lane graph "a measured no-op" — the two files disagree |
| `diag_planner_holds.py` | `PLANNER_HOLDS_GUARDED` 4/4 | ⛔ **the headline arm CANNOT GO RED.** Arm 1 is a hand-written loop *inside the test* that increments its own counter to its own constant; it never calls `compute_planner_trajectory` and never reaches the `raise` at `planner.py:163-169`. **Delete the guard from `planner.py` and this arm still passes.** Arm 3 is `1 < 20` on two constants the test just assigned. Arm 2 and arm 4 are genuine |
| `diag_scorer_components.py` | `SCORER_COMPONENTS_CONFORM` 18/18, 51 s | the **best** file in the package, and it has three holes. **CAN GO RED:** D5 (real `aggregate` call), D3 comfort-varies, D4 stopped/DDC-fires, and ⭐ **the derived-speed-vs-log arm, the strongest control here** (independent of `TRAJ_DT`; measured 4.524 vs 4.523). ⛔ **CANNOT GO RED:** both EP arms assert the diagnostic's **own** `max(0, min(1, …))` clamp — `lon x1.5` really advances **1.50×** the teacher and is printed as `1.000` because the test clamped it, so the arm named *"EP does NOT reward overspeed"* cannot observe overspeed; "progress spans the REAL horizon" asserts `max|advance| > 5.0`, which the **old 2-point behaviour would also pass** (its advance was the largest, not the smallest); the "old logit sum" control is arithmetic on the test's literals; the dead-code AST arm inspects **top-level statements only** and is blind to a nested early `return` — the normal shape of the defect. ⚠️ And **D3 asserts "comfort VARIES", which is a narrower question than "comfort RANKS"** — see R3-3 |
| `diag_schedule.py` | `SCHEDULE_OK` 5/5 | ⚠️ **CANNOT GO RED for REFe.** It builds its **own** `AdamW` + `CosineAnnealingLR` and never imports `train.py`; it tests PyTorch. The `--no-cosine` "control" is `sched = None`, which is tautological. Change `eta_min` in `train.py` and this still prints `SCHEDULE_OK`. *(It is a faithful retyping of `train.py:385`, so the fact it encodes is right.)* |
| `score_proposals.py --selftest` | `SCORER_SELFTEST_OK` | reduces to "six imports succeeded". Two of its five printed facts (`agent 0 is the ego`, `polygon cache exists`) **never set `ok=False`**. ⚠️ `:595` — `sys.exit(selftest() if a.selftest else selftest())` — **the `--selftest` flag is a no-op**. ⛔ And the run **prints stale instructions**: `COMPONENTS` labelled *"the order the scorer target must use"* is the tuple `train.py:117-125` documents as **mis-mapped**, and the closing lines still say *"NOT DONE YET: wiring these targets into train.py"* |
| `diag_batch.py` | ⛔ **exit 1** | crashes at line 76 on `NuPlanTTC.info` shape `(8,128,5)`; the verdict line is **never reached**. 2 of 6 calculators raised `IndexError` and the failures were swallowed by a bare `except`, so a run in which *all six* crashed would print `BATCHED PATH AGREES` |
| `diag_keys.py` · `diag_curbdist.py` · `diag_aimed.py` · `diag_goal.py` · `diag_determinism.py` · `diag_cost.py` · `diag_enrich_cost.py` | all exit 0 | **pure printouts — no assertions, no verdict token, CANNOT GO RED by construction.** `diag_goal.py` contains a genuinely discriminating control (onto-goal 1.0 vs teacher 0.0) that is never asserted. `diag_determinism.py` samples 5× **within one process**, which is the half that was never the defect (its own docstring asks about *between* runs) |
| **my `r3_probe.py`** | §A–E | controls: SELF `0.00000`, JITTER `0.00014`, and a known-real bad input (BGR/RGB, `0.19360`) for scale — the normalisation finding is 2.9× the latter |
| **my `r3_scorer.py` / `r3_scorer2.py`** | §1–4 / §A–B | the NPC arm's control is the log-replay splice, which **could have shown zero difference and did not**; the comfort arm's control is the savgol-only aggregate, which **could have matched the banked value and did not** |
| **my `r3_kin.py`** | §1–3 | mutation arm at `TRAJ_DT = 0.1` goes to a **1.94× error**, so the 2.4 % agreement is meaningful |

⭐ **The pattern across the package.** Of ~60 assertions in 14 instruments, the ones that can
genuinely go red are the ones that call the shipped code and compare it to something derived
**independently** — live forward hooks, `data_ptr` identity, the simulation log's own kinematics, a
map-derived over-curb candidate, a real `aggregate` call. The ones that cannot are the ones whose
expected value is an expression over the code under test, or arithmetic on the test's own literals.
**That is exactly the failure class the package's own CLAUDE.md names**, and it has reproduced here
nine times. ⚠️ The most consequential instance is that **the rotary encoding — the defect that
blocked review 2 — is still guarded by an arm that passes for any rotation and an instrument that
cannot fail.** Its correctness has now been established twice, both times by an *external* probe.

---

## 5 · Stale documentation

Audited exhaustively against source; the full per-line table is in this review's working notes. The
figures below are MEASURED at 21:27–21:50.

**Ground truth:** `vitl16` / 1 camera → **329,664,066 / 26,584,642 / 8.064 %**. `vitl16` / 4 cameras
→ **335,611,458 / 32,532,034 / 9.693 %**. ViT-S/1cam → 33,389,762 / 11,802,818. ViT-B/1cam →
104,189,762 / 18,548,546.

1. ⛔ **The fidelity cross-check has inverted, and three documents quote it as proof the
   architecture is right.** `REFE_MODEL.md:53-58`, `REFE_PLAN.md:137`, `POD_HANDOFF.md:50` all say
   the 4-camera reconstruction lands at **5.52 %** against the published **5.49 %** — *"within 0.03
   percentage points … an independent check that the architecture is dimensionally right"*. It is
   now **9.693 %**: a gap of **4.20 points**, and trainable params **+75.1 %** over 18.58 M rather
   than +4.1 %. Its cited artifact `raw/refe_param_crosscheck.txt` is itself pre-change. **This is
   the single sentence most likely to be quoted into a published result, and it is the one the PI's
   accepted decision broke.**
2. ⛔ **Three headline parameter counts are live simultaneously: 316.9 M, 319.0 M, 329.7 M.**
   `POD_HANDOFF.md:49` (316.9/11.88) is what a pod operator reads first. `REFE_PLAN.md:128` (correct)
   and `REFE_PLAN.md:146` (319.0) contradict each other **14 lines apart in one file**.
   `model.py:18, 48, 64` carry 316.9 M in the source's own docstring. The sub-300M overshoot is
   given as "16.9 M" and "19.0 M"; it is **29.66 M**.
3. ⛔ **`REFE_MODEL.md:26-27`'s backbone-choice table is wrong by 10–98 %** (ViT-S 30.2/7.92 vs
   33.39/11.80; ViT-B 96.5/**9.35** vs 104.19/**18.55** — the trainable count is off **2.0×**). That
   table is the record of a PI decision about which backbone to buy.
4. ⛔ **`score_proposals.py`'s docstring (`:28-29`) and `selftest()` (`:586-587`) tell the reader the
   scorer is not wired.** It is (`train.py:339-358`, `:417-420`, bank path is the CLI default). And
   `selftest()` prints the **mis-mapped** `COMPONENTS` tuple labelled *"the order the scorer target
   must use"*. Anyone building a target file from that output produces a silently mislabelled bank.
5. ⛔ **`planner.py`'s docstring (`:21-24`) describes the retired rule** ("highest summed score" —
   that is `select="logitsum"`, kept only as a control) and its aggregation comment contradicts its
   own constant (R3-13).
6. ⛔ **Operational figures a pod would be provisioned on are from a different architecture.**
   `REFE_MODEL.md:158-160` / `POD_HANDOFF.md:60` give peak memory **3.26 GB (batch 1) / 5.36 GB
   (batch 2)**; `validate_model.py` MEASURED **9.58 GB at batch 2** — on an 8.0 GiB card, i.e. it is
   already spilling. `POD_HANDOFF.md:15`'s 0.780 s/sample → 608 A40-h → 25.4 days rests on the same
   pre-change measurement. **Re-measure; do not adjust.**
7. ⛔ **Retracted claims are still being asserted.** `POD_HANDOFF.md:102` cites the **pooled**
   variance guard that `REFE_MODEL.md:245-251` and `train.py:211-217` both retract as inert.
   `REFE_MODEL.md:272-276` and `POD_HANDOFF.md:102-107` quote `goal_reaching`'s 45/74-vs-13/74
   ordering as *proof it works*; `train.py:120-122` cites **the identical numbers** as the reason it
   was removed — it rewarded overspeed. Same figures, opposite verdict, docs carry the favourable one.
8. ⚠️ **Fixed things still described as broken.** `REFE_MODEL.md:260, 264-267` says comfort varies in
   **0.0 %** and "a sixth of the scoring head does no work"; measured on the live bank it varies in
   **85.6 %** of frames (⚠️ though see R3-3 for *why* it varies). `REFE_MODEL.md:155, 145-146, 15`
   and `POD_HANDOFF.md:97-99` all say the targets are not built or not wired.
9. ⚠️ **Counts that have drifted:** the candidate set is **11** (`jerky` and `reverse` are missing
   from every doc that lists it, which say 9); `POD_HANDOFF.md:78` says "Ship: `refe/` (7 files)" —
   there are **26** `.py` files; the target bank is **2,182** tuples over **10** logs, not the
   1,746/8 in `POD_HANDOFF.md:36, 54`; `README_DIAGNOSTICS.md:26` (654 frames) and
   `POD_HANDOFF.md:101` (982) disagree about the same quantity; `train.py:87` says "the set is 8-9
   today".
10. ⚠️ **Omissions that read as completeness.** `REFE_MODEL.md` §4's controls table and
    `REFE_PLAN.md:152-155` both predate the largest correctness fix of the round — the random
    `self.pos` table replaced by axial RoPE. `README_DIAGNOSTICS.md` documents 12 of 13
    `diag_*.py`; the one it omits is **`diag_rope.py`**, the instrument for that fix. And
    `README_DIAGNOSTICS.md:43`'s "trajectory decoder context 16 vs scoring decoder 128" is true only
    at the diagnostic's shrunken 128×256 input; at production it is **16 vs 1,920**, and the scope
    is not stated.
11. ⚠️ `README_DIAGNOSTICS.md:26` quotes `diag_curbdist.py` numbers (10.66 → 19.06 m at +14 m,
    minimum 0.87 m at +40 m) that **do not reproduce** on the log I ran (1.44 m at +14 m, 22.25 m at
    +40 m). The instrument is log-specific and the README names no log.
12. ⚠️ **A correction to review 2, for the record.** Review 2 §3.2 states that DINOv3's `"separate"`
    normalisation *"makes the two axes span the same angle regardless of aspect ratio"*. MEASURED:
    the span is `2 − 2/N` per axis, so at an 8×16 grid the axes span **10.996** and **11.781** rad —
    not equal; they converge only as the grid grows (at 32×60: 12.174 / 12.357). The verdict is
    unaffected — the fix is bit-identical to the reference — but the stated mechanism was an
    overstatement. **And review 1's "weight decay: not stated" is refuted by Table A12** (R3-8).

---

## 6 · What I could not do

* ⛔ **I could not verify the DINOv3 reference formula from a primary source on this box.**
  `transformers`, `timm` and the `facebookresearch/dinov3` tree are all **ABSENT** from the venv and
  from D:, and installing into that venv is the operation that previously replaced torch with a
  wheel the driver could not run. What I verified is that `build_axial_rope` and `apply_rope` are
  **bit-identical** to the reference formulation that `diag_rope.py` encodes, at four geometries,
  and that the resulting trunk output matches that reference **exactly** on the real checkpoint.
  **The reference formulation itself remains INHERITED from review 2.** ⭐ The cheapest thing that
  would settle it: bank `rope_position_encoding.py` from the DINOv3 release into
  `TanitAD Research Lab/Library/` and have `diag_rope.py` import it, so the comparison is against a
  file we hold rather than against a retyping.
* ⚠️ **The DAC aggregation loss (R3-6) is confirmed as a mechanism and measured as an exposure
  (9.4 % of rows), but I did not observe an actual loss** — n = 1 frame scored per prefix.
* ⚠️ **The frozen-NPC measurement (R3-2) is n = 12 frames, endpoint-only**, on one run. Direction and
  selectivity are established; the magnitude under the full rollout is not.
* ⚠️ **No trained checkpoint exists**, so nothing about loss curves, convergence, or the scorer's
  actual influence on selection is measurable, by me or by anyone.
* ⚠️ Every bank number is a **read of a growing artifact** and carries its stamp.

---

## 7 · Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| **This review** | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/REVIEW_3_FULL.md` — **staged** | no |
| R3 architecture / RoPE / normalisation probe (the R3-1 evidence) | `scratchpad:r3_probe.py` + `r3_probe.out` | **YES — scratchpad only** |
| R3 per-prefix scorer probe (aggregation, carry isolation, frozen NPCs, seed) | `scratchpad:r3_scorer.py` + `r3_scorer.out` | **YES — scratchpad only** |
| R3 NPC-future / comfort-branch experiment (the R3-2 and R3-3 evidence) | `scratchpad:r3_scorer2.py` + `r3_scorer2.out` | **YES — scratchpad only** |
| R3 kinematics-vs-log cross-check with the `TRAJ_DT` mutation arm | `scratchpad:r3_kin.py` + `r3_kin.out` | **YES — scratchpad only** |
| Extracted paper text (301,223 chars, PRIMARY) | `scratchpad:paper_text.txt` | **YES — scratchpad only**; regenerable from `C:/Users/Admin/dz/DriveZero/drivezero_report.pdf` |
| Raw instrument outputs, all 17 runs | `scratchpad:refe_diag_out/*.out` | **YES — scratchpad only** |

⚠️ **The four probe scripts are deliberately not staged**, for the same reason review 2 gave: they
measure a package under active development and banking them would freeze a snapshot of a moving
target. **Every number they produced is reproduced inline above with its file:line and its stamp.**
⭐ **If the PI wants one kept, it is `r3_probe.py`'s section C** — the image-normalisation arm. It is
the only instrument in the programme that can see R3-1, its right home is inside `refe/diag_rope.py`
beside the RoPE arms (same trunk, same question, same controls), and `diag_rope.py` needs
assertions anyway.

⛔ **No code was changed by this review. No file in `refe/` moved during it** (md5s identical at
21:22:48 and 21:50:03). Every minimal fix above is described and not applied.
