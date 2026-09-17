# RL Stage A — the lever and its control ran, the lever is WORSE, and the damage is in SELECTION. ⛔ The REPLICATE did not run, so this is still not a lever claim.

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: T0** (deployed sampler, recorded
future, proxy reward) · **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000,
seed 0 · **2 of 3 Stage A arms complete:** `L1-RL-s0` ✅ · `L1-NORL-s0` ✅ · `L1-RL-s1` ⛔ **FAILED
TWICE (CUDA OOM)**.

## ⛔⛔ Read this before any number below

Nine of twelve paired metrics are **separated** on the programme's own estimator. ⛔ **That is
NECESSARY, NOT SUFFICIENT, and this rig is the reason the rule exists.** A separated interval from
**one seed per arm** answers *"would another draw of episodes say this?"* — never *"would another
training run say this?"*. MEASURED elsewhere in this programme: a replicate at the **same seed**
cleared "separated" on **14.3 %** of cells on the v7-tiny rig and **55.6 %** on WP-D. The replicate
is what bounds that floor **and it has not completed**.

⇒ **No statement below is a claim about the RL lever.** They are absolute, controlled, paired
differences between two arms that differ in one variable, awaiting their floor.

## ⭐ The one-variable proof — MEASURED, not argued

A flag in `argv` is a claim about the **launch**. These are the trainer's own step-0 rows, which is a
claim about the **run**. Of **44** scalar fields logged at step 0, **35 are BIT-IDENTICAL** across
the two arms and **9 differ** — and every one of the 9 is either wall-clock or downstream of the RL
term:

| differing field | `L1-RL-s0` | `L1-NORL-s0` | why |
|---|---|---|---|
| `rl_coef_abs_sum` | 0.438634 | **0.0** | the lever itself |
| `frac_nonzero_adv_used` | 0.236111 | **0.0** | the lever itself |
| `rl_part` | +0.18136519 | **−2.2726e−10** | the lever itself (float noise in the control) |
| `loss` | 0.24668123 | 0.06531604 | IL + `rl_part` |
| `grad_norm` · `grad_norm_clipped` | 29.0603 | 3.0445 | downstream of `loss` |
| `param_delta_norm` | 0.010238 | 0.010108 | downstream of `grad_norm` |
| `fetch_s` · `wall_s` | 0.905 · 3.5 | 1.488 · 6.1 | wall-clock, not state |

⭐ **The decomposition closes to the last bit.** `norl_loss + rl_part_RL = 0.2466812345720652`
against `rl_loss = 0.24668123479932547` — a residual of **2.2726e−10**, which is *exactly* the
control's own `rl_part`. So `rl_loss − norl_loss = rl_part_RL − rl_part_NORL` identically: the RL
term is precisely the added component and nothing else moved.

⭐ **And everything upstream is bit-identical** — `il_matched_anchor_m` 0.6531603753566741,
`chain_endpoint_spread_m` 32.89187240600586, `reward_mean` 0.5934856534004211,
`frac_positive_after_bar` 0.01655982993543148, `human_pdms_mean` 1.0 and the four `cand_*_mean` —
which is what proves same seed, same data order, same init. This is the admissibility evidence for
the comparison; without it "one variable" would be an assertion about intent.

## The two arms that ran

Both 600/600 steps, `check_arm_l1.py` exit **0**, `problems: []` — I1/I3/I7–I10 all hold.
`train --arm rl --seed 0 --steps 600 --batch 4 --il-form matched --grad-clip 100` and the
same line with `--arm norl`, on the RTX 4060.

| instrument | **L1-RL-s0** | **L1-NORL-s0** (length-matched control) |
|---|---|---|
| rows · form · clip · release | 600 · `matched_anchor` · 100.0 · `is_release: false` | 600 · `matched_anchor` · 100.0 · `is_release: false` |
| reward first → last | 0.5935 → **0.8884** | 0.5935 → 0.8411 *(first identical — same seed)* |
| IL distance first → last | 0.6532 → **0.4476** m | 0.6532 → **0.4220** m |
| chain spread first → last | 32.89 → **24.24** m | 32.89 → **30.59** m |
| `grad_norm_max` · `steps_clipped` | **216.78** · **3 / 600** | 79.71 · **0 / 600** |
| `human_nc_eq_1_frac_pooled` | 1.0 | 1.0 |
| wall | 1,883 s (31.4 min) | 2,061 s |

⭐ **The clip behaved as Amendment A-1 predicted.** At 100 it bound **3 of 600** on the RL arm and
**0 of 600** on the control — a spike guard, not a rescale. A-1 was made because the original 1.0
would have bound **600/600**, an every-step 17–62× rescale rather than a divergence guard. These are
the first arms run since that amendment and both are consistent with the reason for it.

## Held-out T0 read — 493 windows, stride 10, the SAME windows for both arms

| metric | `L1-RL-s0` | `L1-NORL-s0` |
|---|---|---|
| `sel_pdms` — what selection picks | **0.8712** | **0.9161** |
| `human_pdms` | 0.9860 | 0.9860 |
| ⭐ `fan_pdms_best` — best candidate **present in the fan** | **0.9949** | **0.9938** |
| `fan_pdms_mean` | 0.6992 | 0.6629 |
| `sel_ade_m` | **2.5399** | **2.2601** |
| `sel_fde_m` | 7.5608 | 7.1233 |
| ⭐ `fan_minade_m` | **0.8175** | **0.9051** |
| `fan_endpoint_spread_m` | 28.53 | 35.36 |
| `fan_nc_fail_frac` | 0.1221 | 0.1506 |
| `sel_nc` · `sel_ttc` · `sel_ep` · `sel_comfort` | 0.9260 · 0.8418 · 0.8842 · 0.9980 | 0.9493 · 0.9087 · 0.9142 · 0.9980 |
| `traj_matches_fan_sel` FALSE | **0 of 493** | **0 of 493** |
| selection gap `fan_pdms_best − sel_pdms` | 0.1237 | 0.0777 |
| selection ratio `sel_ade / fan_minade` | **3.107×** | **2.497×** |

## Paired held-out deltas — ALL TWELVE metrics, none dropped

⛔ Reported complete rather than filtered: a table showing only the separated rows is a table chosen
after seeing the data.

| metric | RL − NORL | 95 % CI | |
|---|---|---|---|
| `sel_pdms` | **−0.0449** | [−0.0712, −0.0228] | **separated WORSE** |
| `sel_ade_m` | **+0.2798** | [+0.1129, +0.4634] | **separated WORSE** |
| `sel_fde_m` | +0.4374 | [−0.1667, +1.0849] | **not separated** |
| `sel_ttc` | −0.0669 | [−0.1041, −0.0331] | separated worse |
| `sel_nc` | −0.0233 | [−0.0450, −0.0064] | separated worse |
| `sel_ep` | −0.0301 | [−0.0509, −0.0134] | separated worse |
| `sel_comfort` | **0.0000** | [−0.0060, +0.0059] | **not separated** |
| **`fan_endpoint_spread_m`** | **−6.8289** | [−7.5891, −6.0245] | **separated NARROWER** |
| `fan_minade_m` | −0.0877 | [−0.1414, −0.0336] | separated *better* |
| `fan_pdms_mean` | +0.0363 | [+0.0243, +0.0475] | separated better |
| `fan_nc_fail_frac` | −0.0285 | [−0.0408, −0.0159] | separated better |
| ⭐ `fan_pdms_best` | **+0.0011** | [−0.0005, +0.0034] | **not separated** |

## ⭐ The shape of it, which is what matters

**The fan narrows and the selected plan gets worse, while the best candidate in the fan is
unchanged.** L1 is *mode-preserving* IL — preserving the fan is the entire point of the lever — and
on this arm the fan is **6.83 m narrower** than its control's.

⚠️ **Read the "better" rows honestly: they are all narrowing in disguise.** `fan_minade_m`,
`fan_pdms_mean` and `fan_nc_fail_frac` improve exactly as a fan concentrated toward the mode would:
pull the candidates in and the *average* candidate gets closer and safer while the *best* one does
not (`fan_pdms_best` **not separated**). Every metric that reads better is a mean over a tightened
distribution; every metric that reads worse is what the car would actually execute.

⭐ Three independent readings now point at **SELECTION**, not the generator:

1. **This arm, read against itself:** `fan_pdms_best` **0.9949** against `sel_pdms` **0.8712** — a
   gap of **0.1237** — and `sel_ade` **2.54 m** against `fan_minade` **0.82 m**, a **3.107×** ratio.
   ⭐ This needs no cross-arm comparison and no control: it is one arm read against itself. ⚠️ It
   holds in the control too (**0.0777**, **2.497×**), so it is a property of the architecture, not
   of the lever. ⛔ **But see the next section: "the selector does not pick the good plan" is the
   WRONG reading of this gap, and I wrote it before measuring the random-pick control.**
2. **D3, same model family, same night:** the planner **attends** to the lead (T-G 1.92×, separated
   at all four layers) yet greying the lead out moves the time gap by **0.09× the noise floor**. The
   read is there and unused.
3. **Here:** the RL term narrows the fan **without improving its best member**. It is removing
   options, not making better ones.

## ⛔⛔ CORRECTION — the selector is NOT failing to pick the good plan. The regret is a TAIL.

⚠️ **This section corrects the first version of this document, which read the
`fan_pdms_best` − `sel_pdms` gap as "the fan contains a near-human plan and the selector does not
pick it".** That sentence is an inference from a gap, with **no control**. The control was already
sitting in the banked rows and costs zero GPU: `fan_pdms_mean` is the expected score of a candidate
drawn **uniformly** from the fan, i.e. exactly what a selector that reads **nothing** would score.
MEASURED, paired episode-cluster bootstrap, n_boot 2000:

| | `L1-RL-s0` | `L1-NORL-s0` |
|---|---|---|
| pick-at-random control (`fan_pdms_mean`) | 0.6992 | 0.6629 |
| **the deployed selector** (`sel_pdms`) | **0.8712** | **0.9161** |
| oracle ceiling (`fan_pdms_best`) | 0.9949 | 0.9938 |
| selector **−** random | **+0.1720** [+0.1308, +0.2136] | **+0.2532** [+0.2119, +0.2964] |
| selector **−** oracle | −0.1237 [−0.1709, −0.0835] | −0.0777 [−0.1140, −0.0454] |
| normalised skill (0 = random, 1 = oracle) — mean · median | 0.6680 · **0.9475** | **0.8156** · **1.0000** |
| picks the fan's best candidate **exactly** | 174 / 493 = **35.3 %** | 258 / 493 = **52.3 %** |
| … within 0.01 of it | 229 / 493 = 46.5 % | 329 / 493 = **66.7 %** |
| picks **worse than chance** over its own fan | 63 / 493 = **12.8 %** | 34 / 493 = **6.9 %** |
| share of all regret carried by the worst 10 % of windows | **68.6 %** | **86.5 %** |

⭐ **On the control arm the selector picks the single best candidate in the fan on the MEDIAN
window** (normalised skill median **1.0000**, exact pick on **52.3 %**), and it beats the
random-pick control by **+0.2532** with the interval clear of zero. A selector that "does not pick
the good plan" cannot do that. ⇒ **the earlier sentence is withdrawn.**

⭐⭐ **What is true instead, and it points somewhere much cheaper: the loss is a TAIL.** On the
control, **86.5 % of all selection regret is carried by the worst 10 % of windows** — 49 of 493 —
while the other 444 are at or near the ceiling. That is not "rebuild selection"; that is **find what
those windows have in common**, which is a stratification question answerable on the banked rows at
zero GPU.

⭐⭐ **And it sharpens the RL attribution to something the fan-width story could not say.** The RL
arm hands its selector a **strictly easier problem** — its random-pick control is **higher**
(0.6992 vs 0.6629, because the fan is tighter) and its oracle ceiling is **no lower** (0.9949 vs
0.9938) — and the selector still scores **worse** (0.8712 vs 0.9161). Exact picks fall by **84
windows (−17.0 pp)** and windows selected **worse than chance nearly double (34 → 63)**. ⇒ **the RL
term's damage is localised to SELECTION, not to the generator**, which no reading of
`fan_endpoint_spread_m` alone could establish.

⚠️ **Scope, honestly.** The *within-arm* rows (selector vs its own random control, exact-pick
counts, the regret tail) are absolute readings of one arm against a control computed on the **same
windows from the same fan**, so the one-seed caveat at the top does not touch them. The
*cross-arm* rows (−17.0 pp, 34 → 63) are RL-minus-NORL differences and are **subject to it in
full** — they await `L1-RL-s1` like every other cross-arm number here.

Banked: `raw/selection_skill_vs_random.json`.

## ⭐⭐ THE NEXT LEVER, MEASURED AND PRICED: 5 % of windows carry 62 % of the oracle gap

The tail above is not a mood, it is a **named sub-population**, and it is small. Stratifying the
worst 10 % against the other 444 windows — **reproduced independently on both arms**, which is the
only reason it is quoted at one seed:

| | tail (49) | rest (444) | ratio |
|---|---|---|---|
| `fan_nc_fail_frac` — share of the FAN that collides | **0.3851** | **0.1247** | **3.09×** |
| `fan_endpoint_spread_m` | 31.50 | 35.79 | 0.88× |
| `fan_pdms_best` — was a good plan present? | **0.9728** | 0.9961 | 0.98× |
| `human_pdms` | 0.9405 | 0.9910 | 0.95× |
| `t0` — position in the episode | 73.65 | 74.37 | 0.99× |

*(`L1-RL-s0` reads 0.3377 / 0.0983 = **3.44×**, spread 27.03 / 28.70, `fan_pdms_best` 0.9867,
`t0` 75.98 / 74.11 — same shape.)*

⭐ **It is the CROWDED population, and the fan is NARROWER there, not wider.** `fan_nc_fail_frac`
is a property of the **fan**, not of the selection, so unlike the sub-scores it is not definitional.
And `fan_pdms_best` stays at **0.9728** — a good plan was present even in the tail.

### Which sub-score discriminates — and which is structurally incapable of it

⛔ The tail is defined by `fan_pdms_best − sel_pdms`, and `sel_pdms` is a function of the four PDM
sub-scores, so *"the tail has low sub-scores"* is **partly definitional**. What is **not**
definitional is where each sub-score's failures **live**:

| sub-score | windows imperfect (of 493) | of which in the tail | |
|---|---|---|---|
| `sel_nc` | **25** | **25** | **100.0 %** |
| `sel_ttc` | **45** | **43** | **95.6 %** |
| `sel_ep` | **216** | 34 | **15.7 %** |
| `sel_comfort` | **1** | 1 | — |

⚠️ **`sel_comfort` is imperfect on ONE window in the whole corpus** (sd 0.045). Its absence from the
tail is **structural** — a sub-score that never varies cannot carry regret — so "the selector is not
trading safety for comfort" would have been a wrong reading, and is not made. ⭐ The real
discriminator is the contrast between `sel_ep` and `sel_nc`/`sel_ttc`: **ego-progress failures are
common (216/493 = 43.8 %) and mostly benign**, while **collision and TTC failures are rare and
almost entirely inside the tail**.

### ⛔⛔ And in EVERY collision window, a collision-free plan was sitting in the fan

| | `L1-NORL-s0` | `L1-RL-s0` |
|---|---|---|
| windows whose **selected** plan collides | **25** | **37** |
| …of which the fan still held a **collision-free** candidate | **25 / 25 = 100 %** | **37 / 37 = 100 %** |
| mean share of the fan that was collision-free there | **55.3 %** | **60.6 %** |
| mean `fan_pdms_best` available there | 0.9570 | 0.9824 |

**The selector picks a colliding plan out of a fan that is MAJORITY collision-free, in 100 % of the
cases where it collides.** That is not a hard-scene problem and not a generator problem.

### The repair, priced before any GPU is spent on it

An **oracle-repair ceiling**: lift one named sub-population to the best candidate the fan actually
held, and ask what the corpus mean becomes. ⛔ A ceiling, never a result — no selector reaches an
oracle — and it is equally a basis for **refusing** a lever whose price is too small to matter.

| repair | n | Δ `sel_pdms` | 95 % CI | share of the full oracle gap |
|---|---|---|---|---|
| **collision windows only** | **25** | **+0.0485** | [+0.0228, +0.0801] | **62.4 %** |
| the whole worst-10 % tail | 49 | +0.0672 | [+0.0347, +0.1040] | 86.5 % |
| every window (full oracle) | 493 | +0.0777 | [+0.0454, +0.1140] | 100 % |

*(`L1-RL-s0`: **+0.0733** on 37 windows = **59.3 %** of its larger gap.)*

⭐⭐ **Five per cent of windows carry sixty-two per cent of the recoverable PDMS**, and the
intervention is a **hard constraint, not a learned weight**: never select a colliding candidate
while a collision-free one is in the fan.

⛔⛔ **THE CATCH, AND IT IS THE WHOLE ENGINEERING PROBLEM.** `sel_nc` is scored against the
**RECORDED future** — this is T0. So the table above says what a **perfect collision checker** would
have bought, **not** what the deployed policy can see at inference. A selection-time gate needs a
**predicted** occupancy, and under the vision-only rule it may not read the recorded future. ⇒ this
is not a free re-ranking; it is a **requirement on perception**.

⭐ **Which makes it a direct, quantified argument for the PI's refcv6 perception directive.** The
BEV map head being wired right now is exactly the organ a collision gate would read, and this prices
what it is worth on the selection side: **62 % of the oracle gap**, concentrated in 5 % of windows,
on a rig where the fan already contains the right answer. ⚠️ And it lines up with D3 from the same
night — the planner **attends** to the lead and the read changes nothing — because the tail is
precisely the population where the lead is what matters.

Banked: `raw/selection_regret_strata.json`, `raw/selection_regret_subscores.json`,
`raw/selection_repair_ceiling.json`.

⚠️ **It is still T0** — deployed sampler, **recorded future**, **proxy reward** — so it is a
diagnostic and **never a driving number**. And `fan_pdms_best` is a **max over the fan**, which is
optimistic by construction: an oracle that picks the best of N is not a selector.

## ⛔ Why the replicate failed, twice, and what was NOT done about it

**CUDA out of memory** in `ctx.capture` — the rollout forward — at step 186, then 157, on an 8 GB
card, while another job held **24.85 GB RSS** (pinned-host allocation is where this surfaces). Arms
1 and 2 ran when the box was quiet.

⛔ **The batch was NOT reduced to get it through.** Batch is held constant across arms, so a
smaller-batch arm 3 would not be a replicate of arm 1 — it would be a third condition wearing a
replicate's name, and its "floor" would measure the batch change. It re-runs unchanged when the box
is quiet.

⚠️ **The first failure reported `exit 0`** because the command chain ended in `tail`, whose status
won — the `$?`-after-a-pipeline trap, hit again despite a standing memory about it. **The artifact
check caught what the exit code hid** (`rows 186 != 600`, no checkpoint). The re-run captured the
status directly and reported `ARM3_RC=1` honestly.

## ⭐ One thing the failed arm DID establish

**Seed 1 is the diverging seed, and Amendment A-1 is validated on the arm it was written for.**
`grad_norm_max` **11,511** with **44** steps over 100 in only 157 rows, and the log shows
`g 147.304->clip`. A-1 replaced a clip of 1.0 — which would have bound **600/600** — with 100,
predicting it would bind rarely on stable arms and often on a diverging one. MEASURED: **3/600** on
arm 1, **0/600** on arm 2, **44/157** here.

## Budget

≈1.82 GPU-h spent (2 arms + 2 held-out reads + 2 failed attempts). A completed arm 3 plus its
held-out read adds ≈0.73 h → **≈2.55 h**, above the pre-registered **≤ 2.45 h** condition for
Stage B. ⇒ **Stage B is recorded NOT RUN**, never "unnecessary".

## Where the artifacts live

⚠️ **ONE PLACE:** `devbox:C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/` —

| | |
|---|---|
| `l1-rl-s0/` | `ckpt.pt` **1.3 GB**, `metrics.jsonl` (600 rows), `config.json`, `run.json` |
| `l1-norl-s0/` | `ckpt.pt` **1.3 GB**, `metrics.jsonl` (600 rows), `config.json`, `run.json` |
| `heldout_l1-rl-s0.json` | 493 rows, 211 KB (+ its `.log`) |
| `heldout_l1-norl-s0.json` | 493 rows, 210 KB (+ its `.log`) |
| `l1-rl-s1/` | ⛔ no checkpoint — both attempts died in the rollout forward; `l1-rl-s1.log` kept |

The two run directories in full are
`devbox:C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/l1-rl-s0/` and
`devbox:C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/l1-norl-s0/`.

Too large for the repo; `raw/` here carries every number that has been quoted, and the held-out rows
are keyed by **`sha12`**, never by clip id.

## Next

⭐ **The named next lever is above and needs no GPU to specify: a collision gate on selection, worth a measured 62.4 % of the oracle gap, blocked only on a predicted occupancy the perception wiring is being built for.**

`L1-RL-s1` re-runs **unchanged** — same batch, same flags, same seed 1 — once the box is quiet; it
is the only thing between this package and a lever claim. **Stage B is NOT RUN** under the
pre-registered ≤ 2.45 h condition, and that is recorded as a budget outcome, never as
"unnecessary".
