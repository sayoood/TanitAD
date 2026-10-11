# P6 · the paper-faithful full REFe run on ONE A40 (design spec)

> Written 2026-10-11, design only: no training, no pod. Gated on P2's GO (`eval/PREREG_P2.md`, "Next arm").
> PI 2026-10-11: *"a full run on a new 7a40, only if we have confidence that the result will be better and achieve
> our goals. We keep our encoder."* Clarified the same day: **"7a40" is ONE A40 for about 7 days**, not 7 GPUs.
> Goal bar: **94.55 PDMS** (DriveZero Table A13, DINOv3 ViT-L, navtrain, no SimScale; PUBLISHED, p. 31).
> Pre-registration: `eval/PREREG_P6.md` (DRAFT). Confidence: `P6_CONFIDENCE.md`.
> Evidence classes: MEASURED (ours, artifact named), PUBLISHED (paper), INHERITED (our docs, not re-verified),
> ESTIMATED, UNVERIFIED.

## 0 · The answer on time, first: does 25 epochs fit 7 days on one A40?

**Only with the zero-extra-forward labeller (L1, §3), and with no margin. With any design that adds a forward
pass, it does not: 9.2–13.5 days.**

### 0.1 Reconciling the two clocks (36–39 days vs ~7 days)

The two figures measure different precisions. Neither is a mistake.

| | `TRAINING_TIME.md` §3 row C | the final REFe run (MEASURED) |
|---|---|---|
| precision | **FP32** | **bf16 autocast + TF32 + `torch.compile`**, 2 loader workers |
| source | RTX 4060, batch 1, extrapolated to ViT-L × 4 cameras (3.217–3.372 s/sample), ÷ an ESTIMATED A40 multiplier of 2.475–2.559 | the run's own `metrics.jsonl` on the A40 |
| s per sample-pass | 1.257–1.362 (ESTIMATED) | **0.2227** (57.0 s/step ÷ 256, epochs 0–11) |
| work | 100 K × 25 | **2,579,126 sample-passes, 10,075 steps, 25 scene-epochs** |
| time | 873–946 A40-h = 36–39 days | **168.97 h = 7.04 days** |

**Sources for the measured column:**
- `E:/Projects/TanitAD-artifacts/refe-final-2026-10-01/summary.json`, `metrics.jsonl` and `config.json` (batch 4,
  accum 64, `epoch_unit` scenes, epochs 25, amp bf16, tf32, compile, workers 2).
- Step-to-step deltas: median **59.30 s**, p10 56.92 s, p90 65.72 s, summing to **168.97 h**.
- Started 2026-09-23T23:17:34Z and finished 2026-10-01T00:18:49Z, including one resume.
- Peak memory **21.78 GB** at micro-batch 4.

**Row C was right for FP32.** On 2026-09-23 the A40 measured FP32 at batch 4 at **1.282 s/sample**
(`refe/train.py:1070-1072`, `refe/bench_speed.py`). That falls inside row C's predicted 1.257–1.362, so the
ESTIMATED multiplier is now confirmed.

**The ~5× difference is precision plus compilation, not caching.**
- 1.282 ÷ 0.2227 = **5.76×**.
- Of that, bf16 accounts for 3.5× (bench: 1.282 → 0.366 s/sample).
- The remaining 1.64× separates the bench from production: `torch.compile`, TF32 and the loader workers. Which flag
  contributes how much is UNVERIFIED, because the bench's flags were not re-read here.

**No step is cached, and none can be.**
- The LoRA sits in the Q/V projections of every ViT-L block, so each sample-pass in each epoch re-runs the full
  4-camera trunk forward and the LoRA backward.
- `train.py` has no feature cache (`grep` for a feature cache returns 0 hits; the only "precompute" at `:241` refers
  to targets).
- Both clocks count the same 2.58 M sample-passes, each at 256 samples per optimiser step (4 × 64).

**Where the "~7 days" comes from.** It is the final run's own measured 7.04 days for the identical sample count.
P2's 59.6 s/step agrees with it.

`TRAINING_TIME.md` §6 already predicted that bf16 "roughly halves" the time. The measured gain is 5.76×, so that
prediction undershot. ⇒ **ESCALATE:** `TRAINING_TIME.md` §0/§3 needs a superseding note pointing here. I have not
edited it (another document's owner).

### 0.2 P6 on one A40

| quantity | value | class |
|---|---|---|
| bank | 189,888 tuples over **103,037 scenes** | MEASURED, final run `bank` events, epochs 4–27 |
| work | 25 × 103,037 = **2,575,925 sample-passes = 10,075 optimiser steps** at 256 (25 × ⌈103,037 / 256⌉ = 25 × 403: the trainer rounds the partial last batch UP per epoch; corrected 2026-10-11 from 10,063 = the pooled ceiling, which the trainer never runs. The final run's 10,075 is MEASURED) | arithmetic |
| pure training step, no labeller GPU work | **57.0 s** | MEASURED, median of epochs 0–11; epochs 12–27 with the on-policy dump read 60.99 s |
| pure training time | 10,075 × 57.0 s = **159.5 h = 6.65 days** | MEASURED basis |
| checkpoints + egress | +2 % | ESTIMATED; already inside the 168.97 h |
| **L1** (deferred score backward, §3.2): GPU stall ≤ 5 % (gate) | **6.8–7.1 days** | ESTIMATED |
| L1 forced to micro-batch 2 by memory (§3.2) | **6.8–8.2 days** | ESTIMATED; batch-2 per-sample throughput UNVERIFIED |
| **L2** (an extra no-grad prefetch forward): +0.08 to +0.22 s/sample | **9.2–13.5 days** | upper end MEASURED as the on-policy dump's 0.22 s/sample (`raw/2026-09-26-onpolicy-switch/dump_cost_pause_ab.txt`); lower end ESTIMATED (a compiled no-grad forward ≈ ⅓ of a training pass) |
| FP32, as Table A12 states | 2,575,925 × 1.282 s = **917 h = 38.2 days** | MEASURED rate (bench) |

**Largest faithful subset that fits 7 days** (168 h − 2 % = 164.6 h of steps). Each run is complete: the cosine is
re-timed to reach 0 at its own last step.

| design | epochs that fit | expected PDMS cost vs 25 epochs |
|---|---|---|
| L1, stall 0–5 % | **24.6–25** | ≈ 0 |
| L1 at micro-batch 2, +0–15 % | 21–25 | 0 to −0.3 (ESTIMATED) |
| L2, lower bound (+36 %) | **19** | −0.1 to −0.5 (ESTIMATED) |
| L2, upper bound (+99 %) | **13** | −0.3 to −1.2 (ESTIMATED) |

**Basis for the cost column.**
- Its scale is REVIEW_7 §3.6's estimate of the whole schedule defect, +0.3–1.0 PDMS (ESTIMATED), run in reverse.
- No valid shorter-run curve exists. The final run's mid-run snapshots sit part-way down a cosine timed for 10,075
  steps, and they span bank growth and two scorer regimes, so they are a lower bound and not a proxy.
- ⛔ A shorter run is a **declared departure** (Table A12: 25 epochs) and is named in the arm.

⭐ **Recommendation:** L1 at 25 epochs. Its two risks (memory, label latency) are each settled by a launch-gate check
(G-LIVE-MEM, G-LIVE-LAB) before any GPU-day is spent. If G-LIVE-MEM fails at micro-batch 4, drop to micro-batch 2 ×
accum 128 and keep 25 epochs (≤ 8.2 days). L2 is the fallback only if L1 cannot be built.

**PI DECISION P6-T:** accept 6.8–8.2 days for 25 epochs, or cap the run at 7 days with a declared epoch cut.

**Pod time is more than GPU time.** Before step 0, the new pod needs (ESTIMATED 1–2 days, GPU idle):
- the navtrain data re-fetch (164–449 GB, INHERITED);
- the bank restore from HF (8.10 GB);
- the metric-cache build (§3.3);
- the gate.

## 1 · Row-by-row: paper | our P2 | P6

Sources:
- **Paper**: DriveZero (arXiv 2609.06055, banked in the Library) as read in `REVIEW_7_GAP_TO_PAPER.md` §1.3 / Table A12
  / p. 8.
- **P2**: `eval/PREREG_P2.md`.

| variable | paper (PUBLISHED) | our P2 | **P6** | P6 = paper? / reason |
|---|---|---|---|---|
| encoder | DINOv3 ViT-L (Table A13) | DINOv3 ViT-L/16 frozen, rank-32 Q/V LoRA | same as P2 | PI: "we keep our encoder" |
| init | not stated beyond the backbone | warm start from `model_final.pt` | **DINOv3 + fresh LoRA + fresh heads**, as the final run | = paper (a full run) |
| data | navtrain, no SimScale | the `train_grow` bank | the **same** bank: 189,888 tuples / 103,037 scenes, restored from HF `Sayood/tanitad-refe-navtrain-banks` | no re-selection. ⚠️ The parity rule's PhysicalAI corpus is not involved. |
| schedule | 25 epochs over the full set | 3 epochs | **25 epochs over scenes** (`--epoch-unit scenes`, `train.py:1784`), **full bank from step 0**, no `--grow` (`:1742`) | = paper. The final run's growing bank (epoch 0 = 7,536 scenes) was a departure. |
| augmentation | "alternative route intents" | lane rank / goal horizon twins, alternating across epochs | same twins | = paper in kind (method unstated) |
| batch | 256 | 4 × 64 | **4 × 64 = 256** (`--batch` `:1758`, `--accum` `:1766`) | = paper's effective batch; micro-batch invariance probed by `refe/check_microbatch_invariance.py` (result UNVERIFIED here) |
| optimiser | AdamW, lr 2e-4, cosine to 0, no warm-up, wd 0.01, clip 1.0 | lr 5e-5 (warm start) | **lr 2e-4** (`:1776`), cosine to 0, wd 0.01 (`:1778`), clip 1.0 (`:1590`) | = paper |
| precision | **FP32** | bf16 + TF32 | **bf16 autocast + TF32 + compile** | ⛔ declared departure: FP32 is 38.2 days on one A40 (§0.2). Numerics validated by `refe/diag_amp.py`; effect INHERITED negligible (REVIEW_7 §1.3). |
| scorer targets | **PDM targets on its own proposals, every sample** (Fig. 2; p. 8) | PDM targets on fixed bank proposals, 174,912 sets, `--pdm-only` | **online PDM targets on each sample's OWN current proposals, from step 0, lag 0** (§3) | = paper. This is P6's central delta. |
| plans that are scored | NAVSIM simulates each proposal (LQR) | executed plans (`executed_plans8`, repair on) | executed plans, **the same transform at labelling and at inference** (G-EVAL) | = paper |
| detach | only the candidate trajectories (p. 8) | context un-detached (`--scorer-sees-trunk`) | **un-detached context** (`:1774` → `train.py:1092-1094` → `model.py:751`); candidates stay detached (`model.py:746`) | = paper |
| scorer components | 6 (NC, DAC, EP, TTC, C, DDC) | 6 | **6**; SFT-4's 7th lane output is not built | = paper |
| score-loss weight | not located in Table A12 (UNVERIFIED) | 0.1 | 0.1 (`--score-w` `:1794`) | kept from the final run, declared |
| nav input | **navigation command** derived from the teacher's goal point ([56] OpenScene); NAVSIM `driving_command` at test (p. 8, inferred) | two Fourier goal points from the (route-fixed) polyline | **command** (§2 D6); PI decision P6-N | = paper; admissible (PI 2026-09-28: nav commands IN). It removes the route-correction dependency. |
| route correction | inside the DriveRL teacher (bank side) | route fix ON in the planner seam | bank side unchanged; the student has no route input in command mode | = paper |
| twin labels | "the same augmented route is used to evaluate student proposals" (p. 8) | scenario route for every set | **twin route for twin samples** (§2 D7) | = paper; fallback declared |
| yaw loss | L1 over points; nothing about wrapping | wrapped (`model.py:788`), weight 0.1 | **unwrapped L1**, weight 0.1 (§2 D5) | = paper (literal reading); PI decision P6-Y |
| selection at test | argmax of the PDMS-formula score | `navsim_v1` | `navsim_v1` (`planner.py:518`) | = paper |
| seed | – | 0 | 0 (`--seed` `:1796`) | – |
| map / BEV information | – | none at inference | none at inference. A training-only map-supervised aux head (REVIEW_7 P7.1) is **admissible** (PI 2026-10-11) but is NOT in P6. | P6 stays paper-faithful; P7.1 is a separate arm |

## 2 · Code deltas (file:line of today's code; each with an acceptance test)

`refe/train.py` is **not edited here**; another agent implements this spec. Each test must show:
- a **non-zero gradient on the new or changed parameters**; and
- a **control that must read EXACTLY 0.0**, which proves the probe works at all.

Each test also has a deliberate-regression (mutation) arm.

| # | delta | today's code | acceptance test (non-zero / exact-0.0 / mutation) |
|---|---|---|---|
| D0 | config hygiene: `ModelCfg` refuses undeclared attributes | plain `@dataclass` at `model.py:57` | setting an unknown attribute raises; **mutation:** remove the guard → the G-HYG check FAILS |
| D1 | **online labeller**, replacing the on-policy bank | `OnPolicyBank` `train.py:397`; label path `:1479-1526`; labeller `onpolicy_relabel_pdm.py` (`label_log` → `PT.score_plans(PT.metric_cache_parts(sc), PT.executed_plans8(props, repair=True))`) | (a) labels **bit-identical** to offline `onpolicy_relabel_pdm.py` on the same proposals for ≥ 64 sets; (b) score-loss-only gradient on `score_q_mlp` and the scorer heads is **non-zero**; with the labeller off (`:1526`, `l_score = score.sum()*0.0`) it is **exactly 0.0**; (c) **mutation:** feed one-step-stale proposals → identity (a) FAILS |
| D2 | scorer reads its OWN proposals at lag 0 | `model(..., score_extra=cand_xy)` `train.py:1490-1491` scores bank candidates (`sx`) | P6 uses `score` (own proposals, `model.py:746`) with D1 labels and `score_extra=None`. Test: from the score loss alone, `traj_head` gradient **exactly 0.0** (candidates detached), LoRA gradient **non-zero** (p2_gate G2 pattern) |
| D3 | un-detached context | exists: `--scorer-sees-trunk` `:1774` → `:1092-1094` → `model.py:751` | `refe/p2_gate.py` G2 verbatim: un-detached → non-zero LoRA gradient; detached arm → **exactly 0.0** |
| D4 | **deferred score backward** (L1, §3.2) | `loss.backward()` `train.py:1586-1594` runs right after the forward | over one accumulation window, the gradient with deferral is **bit-identical** to the undeferred gradient given the same labels (same accumulation order); **mutation:** shift labels by one micro-batch → mismatch, FAIL |
| D5 | unwrapped yaw L1 | `dy = (dy + math.pi) % (2*math.pi) - math.pi` `model.py:788`; `yaw_w` 0.1 `:792` | analytic: error 2π−ε gives a yaw loss of **(2π−ε)·0.1** unwrapped (ε·0.1 is the wrapped mutation arm, which must FAIL); with `yaw_w` 0 the yaw-channel gradient is **exactly 0.0**. New flag `--yaw-wrap {on,off}`, **named in argv** (its default must not silently decide) |
| D6 | command input | goal encoder: `n_goal_points=2` `model.py:170`, `goal_dim` `:511`, `forward(img, ego, goal, ...)` `:684`; the planner builds a `driving_command` one-hot at `planner.py:184/335/346/662` | new `cmd_embed`: **non-zero** gradient; the goal encoder is not built in command mode (assert absent, so its gradient is **exactly 0.0** by construction). `goal_to_command()` (the OpenScene rule; **exact rule UNVERIFIED**, the implementer reads [56]) agrees with NAVSIM `driving_command` on ≥ 95 % of non-twin navtrain tokens, a bar set now. Twins take the command of their own augmented goal. |
| D7 | twin-route PDM labels | `metric_cache_parts(scenario)` `navsim_pdm_targets.py:87` always uses the scenario route; twin rows carry their parameter in `aug` (`augment_search.py`, docstring l. 34) | for lane-rank twins, build the cache under `route_lane_rank_patch.set_rank(aug)`; goal-horizon twins share the logged-route cache. Test: PDM-Closed's first centerline lane equals the twin's lane; **mutation:** no patch → logged lane → FAIL. ⚠️ Whether PDM-Closed honours the rank patch is UNVERIFIED. **Fallback** (declared): twin samples get trajectory loss only. |
| D8 | full bank from step 0 | `--grow` `:1742`, `--grow-scenes` `:1747` | G-CLOCK: the bank event at step 0 reads 189,888 / 103,037; realised epochs = 25; steps = 10,075; **mutation:** `--grow` → FAIL |
| D9 | metric-cache precompute (`refe/build_metric_cache.py`, new) | `metric_cache_parts` `navsim_pdm_targets.py:87` is deterministic per (token, route) | `score_plans(cached)` == `score_plans(fresh)` **exactly** on 300 tokens; **mutation:** perturb one cached field → FAIL |
| D10 | `refe/p6_gate.py` + `refe/p6_pod.sh` + a `refe_p6` profile in the launch gate | none; `refe/p2_gate.py` G1–G4 is the pattern | §5 |

## 3 · The online PDM labeller

### 3.1 What runs where

**Before step 0 (CPU):**
- D9 caches `metric_cache_parts` once per token, and once per lane-rank twin.
- Its output: the PDM-Closed trajectory, the initial ego state, the interpolated observations, the centerline, the route
  lanes and the drivable-area map.
- These are **training-time map labels, admissible** (PI 2026-10-11). They never reach the model's input.

**Online, in a CPU `multiprocessing` pool:**
- Each worker runs only `executed_plans8(props, repair=True)` → `score_plans(parts, plans8)` on the 64 proposals of
  one sample.
- A set may be split into proposal chunks across workers for latency.
- Workers hold the 4 nuPlan city maps and an LRU of token caches.

**GPU side.** The labeller adds no GPU work.

### 3.2 L1: deferred score backward (recommended)

Within an accumulation window the weights are fixed. So the proposals of micro-batch *k* are exactly those of
*k*'s own training forward.

Loop: `F(k)` (forward with grad) → dispatch the proposals of *k* to the pool → wait for the labels of *k−1* →
`B(k−1)` (wta + BCE, one backward) → `F(k+1)` → …

- **Lag is 0.** Each label is computed on the very proposals its loss is taken on, and no extra forward runs.
- **Latency budget:** one micro-batch period (≈ 0.9 s at micro-batch 4; MEASURED 57.0 s / 64).
- **Memory:** two live graphs. 21.78 GB at micro-batch 4 (MEASURED) ⇒ about 42 GB (ESTIMATED) on a 48 GB card.
  That is tight, so G-LIVE-MEM decides.
- **Fallback:** micro-batch 2 × accum 128 holds two graphs in about the current 21.78 GB.

### 3.3 Throughput, needed vs measured, and the vCPU sizing for one A40

**Needed:** 2,575,925 sets in 6.64 days = **4.49 sets/s** sustained, at 4 sets per ~0.9 s micro-batch.

**Labeller cost per set: no number is admissible yet.**
- About 258 sets/min on 7.65 CPU ⇒ **1.78 CPU-s/set** (`eval/PREREG_SFT4.md` Amendment 3, l. 109; INHERITED). It is
  UNVERIFIED whether that rate included the per-token `metric_cache_parts`, so it is an upper bound for the online part
  only if it did.
- The other banked rates (≈32 s/set teacher-lane; 1.0–1.5 sets/min/worker) belong to **different labellers** and are
  not used.

Needed CPU = 4.49 sets/s × c. Workers sized at 1.5× that, plus latency (L1 must finish 4 sets inside ~0.9 s):

| c (CPU-s/set) | sustained CPUs | labeller workers | **pod vCPU** (+2 loader, +2 trainer/IPC) |
|---|---|---|---|
| 1.78 (INHERITED) | 8.0 | 12–16 | **16–20** (24 recommended) |
| 5 (pessimistic) | 22.5 | 34 | 38–40 |
| 18 | 81 | – | infeasible on one pod: switch to L2 or cut epochs |

The old pod's CPU quota was **7.65** (MEASURED, R32): too small for P6 at any c.

⇒ **Before provisioning, measure c on the dev box CPU** ("M-LAB", ~15 min, 0 GPU): run `score_plans` on cached parts
for 64-proposal sets from navtest, and record CPU-s/set and RSS per worker. RSS per worker is UNVERIFIED, so RAM
sizing waits on M-LAB.

**PI DECISION P6-C:** a pod with ≥ 24 vCPU, final count after M-LAB.

**Reported during the run** (every 100 steps, `metrics.jsonl`):
- coverage, expected to be 100 % minus the 5-s-future guard skips, which are reported and not padded;
- the lag histogram, which must be all 0 under L1;
- GPU stall per step;
- labeller queue depth;
- `op_drift`.

## 4 · HF egress per checkpoint (G-EGR, `refe/hf_egress.py`, fails closed)

| item | size | cadence | total over the run |
|---|---|---|---|
| `ckpt_last.pt` | 228,234,301 B (MEASURED, `raw/2026-10-11-hf-egress/egress_verify_live.json`) | every 6 h | 28 × 0.228 = **6.4 GB** (one HF file overwritten in place) |
| `model_final.pt` | 1,288,484,600 B (MEASURED, same) | once | 1.29 GB |
| 1/64 subsample of the online label sets | ESTIMATED < 1 GB | daily | < 1 GB |

- **Quota:** 199.4 GB of 1 TB private used (MEASURED, same file) ⇒ ample.
- **Integrity:** every push is verified by the listing size, not by the push status.
- **Rate:** the pod → HF rate is UNVERIFIED; G-EGR times the first push.

## 5 · Launch gate: how P6 satisfies the binding rule (PI 2026-09-26)

`stack/scripts/launch_gate.py` is **not on the tip**. The banked copy is
`products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_ckpt/code/fix/stack/scripts/launch_gate.py`.

What that copy already provides:
- an HMAC-SHA256 `PASS_<commit12>.json` bound to profile, commit, `tree_sha256`, `argv_sha256` (`:678`) and the
  data-manifest fingerprints;
- `CHECKS` `:138`, `JOBS` `:148`, profiles `refc/refcv6/refcv7` `:549`.

⇒ **ESCALATE INTEGRATION (Master Mind):**
1. Land `launch_gate.py` on the tip with a **`refe_p6` profile** whose jobs come from a new `refe/p6_gate.py`.
2. `refe/p6_pod.sh` refuses to start `train.py` unless the token verifies against the exact argv and the manifests of
   the bank, metric cache and data.
3. G-DVB in the banked copy imports `tanitad.train.declared_vs_built`. The REFe profile needs its own declared-vs-built
   map, because REFe is not a `tanitad.train` trainer.
4. **Fallback** only if landing slips: a refe-native token with the same schema and key. Never hand-written.

| check | what must hold for P6 | deliberate-regression arm (must make the gate FAIL) |
|---|---|---|
| **G-HYG** | `ModelCfg` / argparse namespace refuse undeclared attributes (D0) | set an unknown cfg attribute |
| **G-DVB** | every P6 argv lever equals the built value: `detach_scorer_context=False`, `yaw_wrap=off`, `nav_mode=command`, labeller `online`, no `grow`, epochs 25, batch 4, accum 64, lr 2e-4, wd 0.01, clip 1.0, amp bf16, `score_w` 0.1, rule `navsim_v1`; and **no map-derived tensor in the model's forward signature** | flip `detach_scorer_context` in cfg only; add an aux-map tensor to the forward |
| **G-LIVE** | real-data smoke of 2 optimiser steps (128 micro-batches, ~2 min): the position, yaw and 6 BCE terms are present and finite; every trainable group (LoRA, traj head, scorer, `cmd_embed`) has a non-zero gradient; D2/D3 controls read exactly 0.0 | drop the BCE term; detach the context |
| **G-LIVE-LAB** | D1(a) identity on ≥ 64 sets; throughput ≥ 1.5 × 4.49 sets/s; L1 stall ≤ 5 %; lag = 0; coverage reported | stale proposals (D1c); a pool of 2 workers must FAIL the throughput bar |
| **G-LIVE-MEM** | L1 peak ≤ 44 GB at the chosen micro-batch, over 200 micro-batches | – (a measurement; on failure, switch to micro-batch 2) |
| **G-CLOCK** | step 0 bank = 189,888 / 103,037; the trainer prints realised epochs = 25 and steps = 10,075; each label set is stamped with the micro-batch id whose forward produced its proposals | `--grow`; stamp mismatch |
| **G-EVAL** | `planner.py` builds the identical model from `ckpt_last.pt`; trajectories and scores are bit-identical on real rows (p2_gate G1); labeller and planner use the identical plan transform (`executed_plans8`, repair on) | repair off in the planner only |
| **G-CKPT** | save → load → resume: model, optimiser, scheduler, sampler position and labeller stamps; 2 resumed steps are bit-identical to uninterrupted ones (`refe/diag_train_resume.py` pattern) | drop the optimiser state |
| **G-SUITE** | clean-tree `refe` tests and `stack` `pytest -q`, 0 regressions | – |
| **G-EGR** | the first `ckpt_last.pt` push is verified by the remote listing size | a wrong expected size |

## 6 · What P6 deliberately does not do

- **No SimScale data**, because we do not have it (REVIEW_7 P7.3).
- **No scorer ensemble** (P7.2) and **no map-supervised aux head** (P7.1). P7.1 is **admissible** as a training-time
  label (PI 2026-10-11) and is the first beyond-paper arm after P6. G-DVB already guards its one rule: the aux output
  never feeds the planner.
- **No warm start from P2.** P6 is the paper's full chain. P2's result tells us which half of the scorer fix carries
  (`eval/PREREG_P2.md`, "Next arm").

## 7 · PI decisions (headline)

| id | decision | recommendation | **PI ruling, 2026-10-11** |
|---|---|---|---|
| **P6-T** | 25 epochs at 6.8–8.2 days, or a 7-day cap with a declared epoch cut (§0.2) | 25 epochs with L1 | ✅ **DECIDED: 25 epochs; 6.8–8.2 days accepted** (PI: *"25 epochs … I accept the 6.8–8.2 days"*) |
| **P6-C** | pod vCPU | ≥ 24, final after M-LAB (dev box, 0 GPU) | M-LAB done (`eval/RESULT_M_LAB.md`): **recommend ≥ 24 vCPU, ≥ 64 GB RAM, ≥ 400 GB disk**; the PI provisions it with the pod |
| **P6-N** | navigation command (paper) vs goal points (P2/SFT-4) | command | ✅ **DECIDED: navigation command** (PI: *"nav command"*) |
| **P6-Y** | unwrapped yaw L1 (paper, literal) | unwrapped | ✅ **DECIDED: unwrapped, as the paper** (PI 2026-10-11: *"yes, unwrapped yaw as the paper"*) ⇒ `--yaw-wrap off` (`refe/train.py`: on = wrapped, today's behaviour; off = the paper's literal L1) |
| **P6-P** | bf16 (forced: FP32 is 38 days) | bf16, declared | implied by P6-T (FP32 cannot fit) |
| **P6-D** | pod data re-fetch (164–449 GB, INHERITED) plus ~1–2 days of setup | approve with the pod | ✅ **DECIDED: data download approved** (PI: *"approve the data download"*) |
| **P6-G** | launch-gate route (§5) | land the banked gate with a `refe_p6` profile (Master Mind) | ⏳ Master Mind |
| **P6-N-SRC** | where the TRAINING command comes from (the PI ruled the command as input, P6-N) | — | ✅ **Master Mind, 2026-10-11, within P6-N:** rank-0 rows use **NAVSIM's own `driving_command`** (`refe/annotate_commands.py`, a pod stage), the exact signal the model receives at test; goal-augmented twins (no NAVSIM command exists for an augmented intent) use the goal rule. Why: train/test parity. The goal rule agrees with NAVSIM's command on only **0.90** of rows (MEASURED, `raw/2026-10-11-p6-impl/command_rule_measure.json`), below the 95 % bar, and train/test inconsistency in this very input already cost +1.05 PDMS once (A9, REVIEW_7 §3.5). A declared departure from the paper's literal 'derived from the teacher's goal'. |

⛔ **NO POD YET, by the PI's own choice (2026-10-11):** *"I did not start the pod. I wait until you give me signal and we have evidence and confidence to surpass the paper."* ⇒ The pod is started by the PI ONLY on the Master Mind's explicit GO signal, and that signal names the evidence it rests on (see `eval/P6_GO_SIGNAL.md`). Deciding P6-T/N/D is not a launch authorisation.
