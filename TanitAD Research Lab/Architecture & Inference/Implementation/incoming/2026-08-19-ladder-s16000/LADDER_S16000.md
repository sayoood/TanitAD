# Trunk progress at v6F-SW-30k@12000/14000/16000 — the ladder cannot resolve it, but the trainer's own metrics can

`MEASURED` · **T0-DIAGNOSTIC — a frozen-trunk linear readout is a world-model
diagnostic, NEVER driving performance** · route A (`unpen`) · seeds 0/1/2 ·
130-clip lead-enriched probe pool, **70 episode clusters**, ⚠️ **not the
40-episode val set** · `taniteval.ci.paired_episode_cluster_bootstrap`,
`n_boot 2000` · ⛔ `overlapping_holdout_se` forbidden.

## 0. The question, and why the obvious instrument could not answer it

The PI asked whether the v6 trunk shows **progress versus the early test**. The
banked 3-seed ladder had four checkpoints whose `n_agents_all` margin declined
monotonically — **+0.262 → +0.243 → +0.217 → +0.211** — which reads as the
latent slowly closing on the trivial control.

⛔ **The withdrawn D1 probe could not be re-used.** It failed its own positive
control, and its headline (*"~1.8 m better than the random null, r +0.159"*) was
an **ego-speed proxy** — 0.02–0.07 m at the eval-optimal alpha, a **25–90×
overstatement**. The comparison therefore ran through the **repaired** ladder,
which carries the trivial-proxy control on every row.

**Sign convention, pinned from `LADDER_3SEED.md:237`:** K1B is
negative-is-better, so `margin = arm_K1B − C-V0_K1B` and **POSITIVE MEANS THE
ONE-NUMBER EGO-SPEED SCALAR WINS**.
⚠️ **`K1B` is the arm's own statistic, NOT the margin.** Read as a margin,
`n_agents_all` K1B falls −1.11 → −1.48 across the series and looks like the
latent steadily improving — **the exact opposite of what the margin says.** This
inversion was caught mid-analysis; `code/margin_series.py` exists so it cannot
be repeated by hand.

## 1. Pre-registration (committed BEFORE any new fit ran)

| outcome | threshold | reading committed in advance |
|---|---|---|
| continues | **+0.13 … +0.19** | a real slow trend; 30k worth measuring properly |
| **flat or up** | **≥ +0.21** | **the decline was noise on a seed-unstable statistic** |
| crosses zero ×3 seeds | < 0 | first genuine positive — provisional until re-run |

## 2. Result — the "flat or up" branch fired, and the series OSCILLATES

| target | s02000 | s09000 | s09250 | s10000 | s11250 | **s12000** | **s14000** | **s16000** |
|---|---|---|---|---|---|---|---|---|
| **n_agents_all** | +0.5816 | +0.2621 | +0.2428 | +0.2173 | +0.2105 | **+0.4249** | **+0.2764** | ⛔ **+0.5102** |
| n_agents_grid | +0.3115 | +0.5693 | +0.2846 | +0.1583 | +0.7324 | +0.1781 | +0.1639 | +0.2943 |
| nearest_any | +0.1618 | +0.2013 | +0.1381 | +0.2076 | +0.1500 | +0.1468 | +0.1289 | +0.1547 |
| lead_gap | +1.8125 | +1.8784 | +1.7430 | +1.7776 | +1.7408 | +1.8332 | +1.7747 | +1.7364 |
| lead_present | +0.0011 | −0.0001 | −0.0001 | +0.0004 | −0.0000 | −0.0001 | +0.0001 | −0.0002 |
| ego_v0 | +4.0636 | +4.3754 | +4.2889 | +4.1627 | +4.2254 | +4.3093 | +4.2637 | **+4.4413** |

⇒ From 11250 the sequence is **+0.210 → +0.425 → +0.276 → +0.510** —
**non-monotone in BOTH directions.** The four-point decline did not continue; it
reversed at the very next checkpoint and then oscillated. It was a wander.

⇒ ⛔ **NO TARGET CROSSES ZERO AT ANY CHECKPOINT.** The 2 048-dimension operative
latent is out-read by one scalar of ego speed on every rung at every step
measured, including the latest. `ego_v0` is at its series maximum.

## 3. ⭐ THE INSTRUMENT'S NOISE FLOOR IS 4.6× THE ENTIRE SIGNAL

| quantity | value |
|---|---|
| between-checkpoint span (steps 9000–16000) | **0.2997** |
| between-checkpoint sd | 0.1152 |
| **within-checkpoint seed spread (mean)** | ⛔ **1.3754** (min 1.2610, max 1.4600) |
| ⇒ noise floor ÷ between-checkpoint sd | **11.9×** |
| ⇒ noise floor ÷ full series span | **4.6×** |

The sign pattern is `-++` at **every** checkpoint from 9000 on — seed 0 is the
only seed ever favouring the latent, and it never separates.

⚠️ **And the two agent-count targets are ANTI-CORRELATED**: from 11250 to 16000
`n_agents_all` rose +0.211 → +0.510 while `n_agents_grid` fell +0.732 → +0.294.
Two views of a genuine scene representation would improve together.

⇒ ⭐ **OPERATIONAL CONCLUSION: this ladder cannot detect trunk progress at this
resolution, and re-running it as specified at 30k would spend GPU on an
instrument whose noise is 4.6× the effect it is asked to measure.** Fix the
power (more seeds, larger pool) or read trunk progress off the metrics in §4.

## 4. ⭐ THE TRUNK *IS* IMPROVING — on its own objective, at FIXED difficulty

`MEASURED` · ⚠️ **TRAINING-LOG metrics, NOT eval output.** Rule 1 of the
operating standard: trainer curves run ~10 % optimistic versus `eval_*.py`, so
these are **not a capability claim** and are not comparable to eval numbers.
They are admissible as evidence that the optimisation is progressing.

| step | `o1_factual_ade` | `o5_growth` | loss |
|---|---|---|---|
| 2,000 | 0.6013 | 0.807 | 1.8950 |
| 10,000 | 1.8611 | 0.303 | 2.7687 |
| 11,250 | 1.3485 | 0.287 | 2.3128 |
| 12,000 | 1.3050 | 0.277 | 2.1761 |
| 14,000 | 1.0980 | 0.236 | 2.0707 |
| 16,000 | 1.0298 | 0.243 | 1.9484 |
| **17,500** | **0.7263** | **0.240** | **1.6724** |

* **Factual ADE is down 61 % from its 10,000 peak** (1.8611 → 0.7263).
* **Rollout error growth has fallen 0.807 → 0.240** — the model's error
  compounds **≈3.4× less** over the horizon. That is a world-model quality
  signal, not a fit-the-mean signal.

⛔ **THE OBVIOUS CONFOUND IS RULED OUT — the task did NOT get easier.** MEASURED
across the same steps: `o5_k` is pinned at **60.000** throughout, `o3_mask_rate`
flat at 0.414–0.438, `o2_tau_s` constant at **2.000**. A falling ADE at a
constant rollout horizon and constant masking is optimisation, not a schedule
artifact.

⚠️ **The step-2000 row is not a counter-example, it is the shape to expect.** Its
ADE (0.6013) is low while its `o5_growth` (0.807) is the series' worst — the
signature of a near-mean predictor that barely moves and therefore barely
compounds. By 17,500 the model reaches comparable ADE with **3.4× better
growth**, which is the meaningful comparison.

## 5. What the two halves mean together

**The trunk is learning something the frozen linear readout cannot detect.** That
is a stronger and more useful statement than either half alone, and it is the
direct answer to the question asked:

* ⛔ **Not claimed:** that the trunk is not improving.
* ✅ **Claimed:** the linear-readout ladder cannot tell, its one apparent signal
  reverted and then oscillated, and the trainer's own fixed-difficulty metrics
  show real improvement over exactly the interval in question.

## 6. Provenance and gates

* **Reproduction gate PASS** — the analyser re-derives all four banked §6a
  margins from JSON on disk to 4 dp (`+0.2621 / +0.2428 / +0.2173 / +0.2105`
  vs banked `+0.262 / +0.243 / +0.217 / +0.211`) **before** reporting any new
  point. An analyser that cannot re-derive the known answer may not be trusted
  with the unknown one.
* **Pool identity** — all three new dumps are byte-comparable to the banked
  pool: `n_frames 5617`, `n_episodes 130`, `max_in_grid_agents_seen 92`,
  `cuda_max_mem_gb 1.457058816`, `latents.pt` **32,630,727 B** — identical to
  `cache_s11250`. Same split, same join file, same 70 clusters.
* **Staleness gate** — `sp1_cache_latents.py` md5 `0b4f4a50…` matches both repo
  copies; `sp2_probe.py` `aabbee36…` matches the parity run's stated md5;
  `pc6_linear_readout.py` and `ll1_ladder.py` verified against the repo AND
  import-probed for the repaired `intercept_col` / `centred` signatures.
* **Checkpoint provenance** — `v6F_sw_step016000.fp16.pt`, md5
  `fccafd8a132dfefb14d4ca8611e5f0b5`, 673,312,891 B, verified identical on Thor
  and the dev box. ⚠️ The **fp16 snapshots**, never the live `ckpt.pt`: the
  trainer rewrites it periodically and reading it mid-write risks a torn file.
* **No load was added to Thor.** Three dumps (~12 min each, 7.7–8.0 fr/s,
  1.46 GB peak) and four fits ran on the dev-box RTX 4060 while S-W training
  continued undisturbed at 26.45 s/step.

## 7. Deliverable manifest

| artifact | where |
|---|---|
| `ll3_s12000.json`, `ll3_s14000.json`, `ll3_s16000.json` | `…/2026-08-19-ladder-s16000/raw/` |
| `margin_series.py` (analyser + reproduction gate) | `…/2026-08-19-ladder-s16000/code/` |
| this report | `…/2026-08-19-ladder-s16000/LADDER_S16000.md` |
| latents for s12000/s14000/s16000 (32.6 MB each) | scratchpad `sp2/cache_s*/` ⚠️ **not banked — regenerable from the checkpoints in 12 min each** |

## 8. Escalation

⚠️ **Next-steps item #2 ("re-run D1 properly at 30k: {oracle, latent, null} ×
n_queries 16 × ≥3 seeds") should not run as written.** At 3 seeds the noise
floor measured here is 4.6× the entire between-checkpoint signal, so the re-run
would return another unreadable number at full GPU cost. Either raise the power
first, or replace the trunk-progress question with the §4 metrics, which answer
it at zero additional cost.

---

# ADDENDUM — step 18,000, and the series closed at 9 points

`MEASURED` · same pool (`n_frames 5617`, `n_episodes 130`,
`max_in_grid_agents_seen 92`, `cuda_max_mem_gb 1.457058816` — identical to every
other point) · same split, same 70 clusters, 3 seeds, route A.

| target | s11250 | s12000 | s14000 | s16000 | **s18000** |
|---|---|---|---|---|---|
| **n_agents_all** | +0.2105 | +0.4249 | +0.2764 | +0.5102 | **+0.4068** |
| n_agents_grid | +0.7324 | +0.1781 | +0.1639 | +0.2943 | +0.3079 |
| nearest_any | +0.1500 | +0.1468 | +0.1289 | +0.1547 | +0.1418 |
| lead_gap | +1.7408 | +1.8332 | +1.7747 | +1.7364 | +1.7695 |
| ego_v0 | +4.2254 | +4.3093 | +4.2637 | +4.4413 | +4.1499 |

## The verdict, now on 9 checkpoints

From 11,250 the `n_agents_all` margin runs
**+0.210 → +0.425 → +0.276 → +0.510 → +0.407** — **up, down, up, down.**
Non-monotone in both directions, four times. It is a wander, and the four-point
"decline" that prompted this whole investigation was noise.

⛔ **NO TARGET CROSSES ZERO AT ANY OF THE NINE CHECKPOINTS.** One scalar of ego
speed still out-reads the 2,048-dimension latent on every rung. The sign pattern
is `-++` at **all eight** checkpoints from 9,000 on, and **seed 0 never
separates**.

| over 8 checkpoints (9,000–18,000) | |
|---|---|
| between-checkpoint span | 0.2997 |
| between-checkpoint sd | 0.1124 |
| **within-checkpoint mean seed spread** | ⛔ **1.3827** |
| ⇒ noise ÷ between-checkpoint sd | **12.3×** |
| ⇒ noise ÷ full series span | **4.6×** |

The instrument's noise floor is unchanged by four extra points, which is itself
the result: **this ladder cannot resolve checkpoint-to-checkpoint differences,
and adding points measures the instrument rather than the trunk.**

## ⭐ And the same day's DINOv3 experiment explains WHY

`E-ACTSTREAM-2` found that on **DINOv3 ViT-L/16 patch fields** all three
predictor arms **beat persistence**, where on these v6 cell fields **nothing
did** across four configurations. Combined with the cell field's measured
**4.5× between/within-episode variance ratio**, the reading is:

> **The v6 cell readout is largely a scene fingerprint, not a dynamics state.
> The ladder has been asking probes to read agent geometry out of a
> representation that barely carries any — and the episode-cluster bootstrap
> correctly clusters away exactly the variance that dominates it.**

⇒ The constraint was the **representation**, not the probes and not the trunk.
That is a better explanation for two independent negative results (D1's
withdrawal and this ladder's non-resolution) than anything about either
instrument.

⚠️ **Unchanged recommendation, now with more evidence:** re-running D1 at 30k as
specified would spend GPU on an instrument whose noise is 4.6× the signal it is
asked to measure. Fix the power, or read trunk progress off the trainer's own
fixed-difficulty metrics (`o1_factual_ade` −54 % from its 10k peak,
`o5_growth` 0.807 → 0.198 at `o5_k` pinned to 60).
