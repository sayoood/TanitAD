# RESULT — refav1 DISTANCE-KEEPING: the cost term the latent can express (`D-REFAV1-DK-COST`)

**Evidence class: MEASURED (ours).** Rigs: dev-box RTX 4060 (the A/B smoke, on the real
step-21109 checkpoint) and CPU (every re-pricing and label statistic). ⛔ **Neither GPU the brief
named was touched** — the A40's refcv5 training and Thor's eval ran undisturbed; the 4060 showed
no python compute before and during.

Raw: `raw/dk_fire_probe.json`, `raw/dk_tau_calib.json`, `raw/dk_direction_full.json`,
`raw/smoke_out_dkw0.json`, `raw/smoke_out_dkw1e-5.json`, `raw/smoke_manifest_*.json`,
`raw/smoke_dkw*.log`, and the three probe scripts beside them.

⛔ **TIER.** The A/B smoke is **T1** (self-action open loop) on **n = 4 windows / 2 episodes** —
far below any capability threshold, and reported as a **plumbing + direction** proof, never as a
driving result. Everything else here is **label arithmetic and re-pricing over banked artifacts**:
no tier, because nothing is rolled.

---

## 0. The one-line answer

⭐ **refav1 can now express "keep distance": the term is in the planner, it fires on 21 of 90
lead-bearing windows of the banked panel, it re-ranks the planner's own baseline set on 21/21, and
on the real checkpoint it broke the constant-velocity degeneracy and moved headway and TTC in the
correct direction.** What is still missing is a **decoded** gap (today's gap is an ORACLE label, so
this is a ceiling) and the **weight sweep with inference-seed replicates** on the full 141-clip
panel, which is GPU-gated and queued below with its exact command.

---

## 1. ⛔ FIRST, A CORRECTION: the brief's premise was FALSE, and the correction is the cheap half

The brief states *"the four-families rule REQUIRES distance-keeping (headway / time-gap / TTC), and
refav1 has never been able to report it — n = 0"*, and names an adapter to `build_lead_tracks.py`
as the work item.

⛔ **MEASURED: refav1 ALREADY reports distance-keeping, and has since 2026-09-04.**
`taniteval/results/RESULT-refav1-21109-openloop.md` publishes, for `cl` / `ha` / `ha0` / `ol`:

| | `cl` | `ha` | `ha0` | `ol` |
|---|---|---|---|---|
| headway (m) / time-gap (s) / min-TTC (s) | **26.2149 / 4.6913 / 22.9028** | 26.9053 / 4.7930 / 24.8532 | 26.1645 / 4.6877 / 22.8378 | 27.0347 / 4.8065 / 25.0674 |

and `refav1-21109-openloop.json` carries `distance_keeping.status = "OK"`, **`n = 87`**,
`n_time_gap = 81`, `n_closing = 39`, with the censoring note and the full per-speed-band
stratification. The wiring that closed it is `taniteval/taniteval/lead_source.py` +
`taniteval/tools/build_lead_block_b1.py` + the banked **B1 eval lead block**
(`…/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz`,
29,556 rows over 147 clips), joined by `(clip_id, RAW frame 2t)` before `analyze`.

⚠️ **Where the "n = 0" came from, because the class matters more than the correction.** It is a
**stale absence claim**, and its source is identifiable: `dump_lead_join.py`'s docstring records
the 2026-08-18 fact that *"val40 ∩ train2308 = ∅"* — true then, and true of the **train** join.
refav1 does not evaluate on val40; it evaluates on the **141-clip v7.2 EVAL split**, for which a
**different** lead block was built on 2026-09-02. Two corpora, one word ("the eval set"). Same
family as the `physicalai` read-set count that rotted four times: **an absence found at ONE
location is not absence**, and a claim about a corpus must name the corpus.

⚠️ **The 433,040-record / 2,308-clip agent join is real and I re-verified it** — it is on this box
at `C:/Users/Admin/tanitad-data/joins/joins/train2400_agents.jsonl.xz`, md5
**`24cbdca8c3b23aafc2fb17e6bf99cf76` == the banked expected value**, 136,689,648 B. But it is a
**TRAIN-corpus** join, and refav1's eval is the EVAL split, so **it is not the input the
distance-keeping family needs** — the adapter the brief names would have produced a join that
cannot be attached to this arm's windows. ⇒ The adapter is **not built, deliberately**, and that is
reported as a decision, not an omission.

⭐ **TWO DIFFERENT `n`s LIVE IN THIS FAMILY AND THEY ARE NOT THE SAME QUANTITY** (`raw/dk_tau_calib.json`):

| n | what it counts | who uses it |
|---|---|---|
| **90** | windows whose **causal lead selection at t0** found an in-corridor vehicle | the **COST**, which is evaluated at t0 before any path exists |
| **87** | windows where the **arm's predicted path** still holds the lead inside the \|lat\| < 2 m corridor for ≥ 1 step | the **METRIC**, which is arm-dependent |

They differ by construction: an arm that steers away loses its lead and drops out of the metric
while the cost still saw one. Reproduced here from the dump (`lead_metrics.distance_keeping` on
`cl` returns exactly **87**, matching the published record). **Quote the cost's n for the cost and
the metric's n for the metric.**

⇒ Under RULE ZERO the refutation is a waypoint. The rest of this document is the lever it freed.

---

## 2. THE DEFECT THE TERM ATTACKS, restated as arithmetic

`refa_v1.plan`'s `_cost_chunk` prices: the goal term, `w_jerk·mean(jerk²)`, `w_kappa·mean(κ²)`, and
`w_vend·(v_end − target_speed)²` **only when `target_speed` is armed — which `refav1_arm.py` never
does**. So on the acceleration channel:

* `mean(jerk²)` is **0 for ANY constant acceleration**, including 0;
* `mean(κ²)` is 0 only at κ = 0 and says nothing about speed;
* `w_vend` contributed **exactly zero cost to every banked refav1 window**.

⇒ **the all-zero control is the joint minimiser, and nothing in the cost prefers any longitudinal
behaviour to any other.** MEASURED on the 21109 panel: `cl` bit-identical to `ha0` on **270/282**,
emitted controls exactly zero on 270/282, κ **identically zero on 282/282**, two distinct plans in
total. ⚠️ And the whole cost landscape is tiny: `plan_cost_cl` over 282 windows runs
**min −0.0 · median 0.0 · p75 1.6e-05 · max 1.41e-04**.

---

## 3. THE TERM

`stack/tanitad/refs/refav1_lon_cost.py` (new, 23 tests) · hooked in `refa_v1.plan` (9 tests) ·
armed from `taniteval/tools/refav1_arm.py` via `--dk-w/--dk-tau/--dk-d0/--dk-gap-source`.

```
cost_dk = w_dk · mean_k( relu( s*(v_k) − gap_k )² )        [cost per m²]
s*(v)   = d0 + tau · v                                      [IDM desired gap, m]
gap_k   = gap0 + v_lead·t_k − s_ego(t_k)
```

⭐ **THE COORDINATE — the question the brief asked.** The term is a function of **exactly two
things**:

1. **`gap0`** — the causal gap at t0, the **one** perception input. M84 measured it is decodable
   from this planner's own `_last_state` latent: **R²_skill +0.3632 [+0.2069, +0.5088]** over 42
   episode clusters, raw-pixel floor **−0.0513**, constant control **exactly +0.000000**, paired
   `field − pix` **+0.4145 [+0.2018, +0.6120] excluding zero**, within-clip **+0.3995** vs shuffle
   −0.0013.
2. **the candidate's own kinematics** — integrated with `rollout_unicycle`'s convention *exactly*
   (step-start speed, `v` updated last, `clamp_min(0)`), pinned numerically against the integrator
   itself, so this cost's gap is the gap `lead_metrics.per_step_gap` later scores rather than a
   second geometry with the same name.

⇒ **it is expressible in `(gap0, own-kinematics)`, and that is precisely the coordinate M84 says
the trunk supplies.**

⛔⛔ **THE ASSUMPTION, NAMED RATHER THAN HIDDEN.** M84's second finding is that the **closing rate
does not decode at all** (`field` +0.0061 [−0.0406, +0.0513]; DINOv3 +0.0114; pixels −0.0001;
nonlinear included; and the cheap fix — an explicit temporal difference — recovers **+0.0145, still
spanning zero**, against a same-breath non-zero control on the gap). So the lead's future speed is
**unobservable**, and the term closes it at `v_lead = v0` (`LEAD_SPEED_STEADY`) — the
maximum-entropy choice, which makes the predicted gap depend **only on the candidate's excess
displacement over constant velocity** and leaves it **exactly `gap0` for a = 0**.
⇒ **This term prices a GAP; it cannot price CLOSING.** The closure travels into the dump
(`goal_rule.distance_keeping_cost.spec.closure_note`) so no reader has to reconstruct it.

⚠️ **Why IDM-shaped and not a pure time gap.** `gap/v ≥ tau` is undefined as v → 0 and would rate a
1 m gap at 0.5 m/s as a 2 s headway — safe by arithmetic, a collision in fact. A pure distance
barrier is speed-blind. **A threshold carries its regime (M74)**: `d0` carries the crawl, `tau`
the cruise.

⚠️ **One-sided, always.** `relu(·)²` is exactly zero once the gap is adequate, so the term can
never reward accelerating into a large gap (that is the target-speed term's job; mixing them would
make the two non-attributable) and it **saturates** rather than rewarding ever-harder braking.

⛔ **Default-OFF and EXACTLY zero.** `w_dk = 0`, or no lead, returns exact zeros — pinned by tests,
each paired with a same-breath control that MUST differ, because a parity test that passes because
nothing ran proves nothing.

---

## 4. P1 — DOES IT FIRE? (`raw/dk_fire_probe.json`, zero-GPU re-pricing of the banked panel)

Population: the 21109 panel's own 282 windows / 141 episodes; states LEAD 90 · NO_LEAD 55 ·
NOT_STRAIGHT 96 · NO_LABEL 41. Gap from the B1 block (**ORACLE**).

| | n | gap0 mean (min) | v0 mean | s\*(v0) mean | shortfall mean (max) | eps |
|---|---|---|---|---|---|---|
| ⭐ **violating** (`gap0 < d0 + tau·v0`) | **21** | 20.56 m (4.74) | **16.44 m/s** | 29.66 m | **9.10 m (27.48)** | 17 |
| **adequate** — CONTROL | 69 | 31.13 m (5.22) | 8.00 m/s | 17.01 m | **0.00 m (0.00)** | 48 |

**Controls read their known values:** adequate windows cost **exactly 0.0** (an identity, not an
estimate); violating windows cost **strictly > 0**; an unarmed spec reads **exactly 0** on every
window.

⭐ **P2 — IT RE-RANKS, ON 21/21.** On the planner's own constant-acceleration baselines both
existing regularisers are **identically zero** (`mean(jerk²) = 0` for any constant `a`; κ = 0 for
both) and `w_vend` was never armed — **so the DK term is the ONLY discriminator and every flip is
attributable to it alone**. At w = 1: cost(a=0) **143.373** vs cost(a=−1.5) **100.034** on the
violating windows ⇒ **21/21 flip from hold-speed to decelerate**.

⚠️ **Violation is a HIGH-SPEED phenomenon**: 16.44 m/s mean against 8.00 in the adequate block.
That is `s*(v)` doing its job, and it is why the term does not fire in the crawl regime that
dominates this corpus's window count.

---

## 5. ⛔ P2's DIRECTION CHECK WAS A NULL — and the null was about the PANEL, not the corpus

The check that is **not** true by construction: does the human decelerate where the term fires?
On the 282-window panel (`raw/dk_tau_calib.json`), GT acceleration from the dump's own `g`:

| tau | n_viol | Δ mean GT accel (viol − adequate) | 95 % CI (episode-cluster) | |
|---|---|---|---|---|
| 1.0 | 10 | +0.0053 | [−0.2635, +0.2794] | spans 0 |
| 1.2 | 15 | −0.2098 | [−0.5272, +0.0751] | spans 0 |
| 1.5 | 21 | −0.2015 | [−0.4455, +0.0446] | spans 0 |

Shuffle controls read −0.0096 / −0.0148 / −0.0012 — the estimator is honest, the panel is small.
⇒ ⛔ **UNDERPOWERED, and reported as such.** 21 violating windows over 17 episodes cannot settle a
direction question. **The eval dump emits 2 windows per episode**; that is a property of
`--window-stride 40`, not of the corpus.

⭐ **P3 — SO IT WAS RUN WHERE IT HAS POWER** (`raw/dk_direction_full.json`). The same question on
the **whole B1 eval lead block**: **8,339 usable LEAD rows over 147 clips at 10 Hz**, GT
acceleration from the block's own per-frame `speeds` (central difference, within clip, contiguous
frames only), clip-cluster bootstrap.

| tau | n_viol | frac | TRUE Δ GT accel (m/s²) | 95 % CI | excl. 0 | within-clip SHUFFLE | ⭐ **EXCESS = TRUE − SHUFFLE** |
|---|---|---|---|---|---|---|---|
| 0.5 | 227 | 2.7 % | −0.3416 | [−0.6240, −0.0452] | ✔ | −0.1481 | −0.1935 |
| 0.8 | 747 | 9.0 % | −0.1597 | [−0.4492, +0.0831] | ✘ | −0.0701 | −0.0896 |
| 1.0 | 986 | 11.8 % | −0.2343 | [−0.5303, −0.0181] | ✔ | −0.0566 | −0.1777 |
| ⭐ **1.2** | 1403 | 16.8 % | **−0.3002** | **[−0.5423, −0.1217]** | ✔ | −0.0881 | ⭐ **−0.2120** |
| ⭐ **1.5** | 2251 | 27.0 % | **−0.2617** | **[−0.4531, −0.0991]** | ✔ | −0.0563 | ⭐ **−0.2053** |
| 2.0 | 3210 | 38.5 % | −0.1732 | [−0.3692, −0.0007] | ✔ | −0.0314 | −0.1418 |
| 2.5 | 4170 | 50.0 % | −0.1958 | [−0.3524, −0.0505] | ✔ | −0.0581 | −0.1377 |
| 3.0 | 5000 | 60.0 % | −0.1982 | [−0.3239, −0.0762] | ✔ | −0.1004 | −0.0978 |

⭐ **THE HUMAN DECELERATES WHERE THE TERM FIRES, and there is a clean interior optimum at
tau ≈ 1.2–1.5** in the leakage-immune EXCESS (−0.2120 / −0.2053). The shipped default
`DK_TAU_TARGET_S = 1.5` sits inside it.

⛔ **THE WITHIN-CLIP SHUFFLE IS NOT ZERO ANYWHERE (−0.031 to −0.148), SO THE RAW DELTA OVER-READS.**
Clips with many tight-gap rows are also clips that are braking overall; the shuffle isolates that
clip-identity component, and **the readable quantity is the EXCESS, never the raw delta**. Same
lesson as M84's agent-count cell, which the shuffle demolished.
⚠️ **tau = 0.8 FAILS while 0.5 and 1.0 pass** — reported as it came out, not smoothed. At 747 rows
over 20 clips the cell is thin and the sweep is not monotone there.

⭐ **The corpus's own time gap says the same thing from the other side** (n = 7,211 rows / 86
clips, v ≥ 1 m/s): **median 2.917 s, p25 1.894 s, p10 1.316 s**; at **v ≥ 10 m/s** (n = 3,218)
**median 2.056 s, p25 1.371 s, p10 0.886 s**. ⇒ **tau = 1.5 s sits at ~p25 of the at-speed
distribution against a median of 2.06 s — it prices the TAIL, not the median, and does not fight
the reference the arm is scored against.**

---

## 6. P4 — THE A/B ON THE REAL CHECKPOINT (dev-box RTX 4060, step 21,109)

Same seed, same flags, **`--dk-w` the only difference**; the 2 dev-box-slice episodes that carry a
violating eval-grid window at `--window-stride 40` (`2172a5a3`: gap 30.29 m at 21.60 m/s;
`2602baaa`: 41.31 / 44.48 m at 27.17 / 27.00 m/s). **n = 4 windows / 2 episodes.**

| `cl` | `--dk-w 0` (control) | `--dk-w 1e-5` (armed) |
|---|---|---|
| ⭐ constant-velocity frac | **1.0000** | **0.7500** |
| ⭐ bit-identical to `ha0` | **4 / 4** | **3 / 4** |
| headway (m) | 34.8689 | **34.9139** |
| time gap (s) | 1.5360 | **1.5381** |
| min-TTC (s) | 24.2072 | **24.3445** |
| speed MAE (m/s) | 0.3337 | 0.6458 |
| along-track MAE (m) | 0.2645 | 0.5101 |
| accel MAE (m/s²) | 0.3595 | 0.6522 |
| ADE (m) | 0.3312 [0.2602, 0.4023] | 0.5615 [0.2602, 0.8627] |
| **lateral cross-track / curvature MAE** | 0.1586 / 0.000414 | **0.1586 / 0.000414 — IDENTICAL** |
| `ha0` (trivial control) ADE | 0.3312 | **0.3312 — unchanged** |

⭐ **THE DEGENERACY BROKE.** `cl` was the constant-velocity control on 4/4 windows and is now 3/4:
**exactly one window moved**, and it is a violating one. This is the first time refav1's
acceleration channel has been priced at all.
⭐ **The direction is right**: headway **+0.045 m**, min-TTC **+0.137 s**.
⭐ **The lateral family is BIT-IDENTICAL** — a same-breath control proving the term touches only the
longitudinal channel, as designed, and `ha0` is unchanged as it must be.

⛔⛔ **WHAT THIS IS NOT.** n = 4 windows over 2 episodes carries **no** family claim. And the ADE
move (**+0.2303 m**) is **BELOW refav1's ≈0.30 m inference-seed floor** — its planner samples, so
an effect smaller than that floor **is not an effect**. The two structural rows (const-velocity
1.0000 → 0.7500; identical-to 4/4 → 3/4) are **exact-equality counts under a fixed seed with one
flag moved**, which is an identity about this pair of runs and not an estimate — those stand; the
metric deltas do not.
⚠️ **The gap is an ORACLE LABEL.** This is a **CEILING** measurement of the cost geometry — the
question M53/M54 left open when a *perfect goal* made this planner **2.03× worse**. It is not a
capability claim, and the dump stamps it (`gap_source: "oracle_label"`, `n_armed`, the join's
`speed_check_max_mps = 6.67e-07` label-free proof).

---

## 7. WHAT IS QUEUED, AND WHY IT IS NOT RUN HERE

⛔ Both named GPUs are busy (A40: refcv5 ~44 h training; Thor: a sibling's eval) and were not
touched. The full panel needs one of them or a long 4060 run (~2.9 h/arm on Thor; the 4060 is
~1.5× slower per window).

**The weight band is MEASURED, not guessed.** The DK cost difference between the two baselines on a
violating window is **43.34 at w = 1**, and the entire existing cost spread is **IQR 1.6e-05**. So
re-ranking begins at `w_dk ≈ 1.6e-5 / 43.34 ≈ 3.7e-07` and is total by ~1e-05.
⇒ **sweep `w_dk ∈ {1e-7, 3e-7, 1e-6, 3e-6, 1e-5}`** — spanning "never re-ranks" to "dominates the
goal" — with `w_dk = 0` as the parity control. The same shape as the `W_KAPPA` sweep that found a
clean interior optimum.

```bash
# Thor, tanitad-train venv, self-contained eval checkout (REFAV1_ARM.md §6 paths).
# One pass per weight; --dk-w 0 FIRST as the parity control (must reproduce the banked
# 21109 panel bit-for-bit, which is the proof the hook did not perturb the shipped path).
R=/home/nvidia/refav1_evalrun
for W in 0 1e-7 3e-7 1e-6 3e-6 1e-5; do
PYTHONPATH=$R/repo/stack:$R/repo/taniteval OMP_NUM_THREADS=6 \
/home/nvidia/venvs/tanitad-train/bin/python taniteval/tools/refav1_arm.py \
  --ckpt   /home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt \
  --config /home/nvidia/experiments/refav1-b1-v72-ep3-speed/config.json \
  --cache    /home/nvidia/data/refav1-fp8-eval \
  --episodes /home/nvidia/data/physicalai-b1-w120-256x640cyl \
  --labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --nav    /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --lead-block    $R/b1_eval_lead_block.npz \
  --dk-gap-block  $R/b1_eval_lead_block.npz \
  --dk-w $W --dk-tau 1.5 --dk-d0 5.0 --dk-gap-source oracle_label \
  --device cuda --window-stride 40 --episodes-n 0 --no-navshuf \
  --dump-dir $R/dk_$W --out $R/refav1_dk_$W.json --arm refav1-21109-dk$W
done
```

⛔ **AND THE SWEEP ALONE IS NOT ENOUGH — IT NEEDS INFERENCE-SEED REPLICATES.** refav1's planner
samples; its seed floor is ≈0.30 m ADE and *"the programme has been comparing single-seed arms"*
(`D-REFAV1-SEED-GOAL-MISMATCH`). ⇒ **the chosen weight and `w_dk = 0` must each be re-run at ≥ 3
`--seed` values**, and any headway/ADE difference smaller than that floor is not an effect. The
`--dk-w 0` parity control doubles as the check that the hook left the shipped path untouched.

---

## 8. WHAT IS STILL MISSING — named, with what unblocks each

1. ⛔ **A DECODED GAP HEAD. This is the one that separates a ceiling from a capability.** Today
   `--dk-gap-source` accepts **only** `oracle_label`, and the tool **refuses** any other value
   rather than banking an arm whose record would misdescribe its own input. M84 measured the gap
   IS linearly decodable (+0.3632 vs a −0.0513 pixel floor), and **decodability is NECESSARY, NOT
   SUFFICIENT (C131)**. *Unblocked by:* a small ridge/MLP head on `_last_state` trained on the
   train-corpus join (which is on this box, md5-verified) and scored on the eval block — no
   trunk training, so it is a dev-box-sized job.
2. ⛔ **CLOSING RATE.** The term cannot price it because the trunk does not carry it (M84 §4/§4b,
   including the refuted cheap fix). *Unblocked by:* the representation work item a sibling owns
   (refcv5 / v7f) — **not** by anything in this cost.
3. ⚠️ **The weight sweep + seed replicates** above. *Unblocked by:* a free GPU.
4. ⚠️ **A curving-candidate gap.** `predicted_along` is straight-line; the error is O(κ²) and
   sub-percent at this planner's `kappa_max` over 2 s, and the metric's corridor gate drops a
   candidate that curves far enough to matter. Stated, not assumed away.
5. ⚠️ **`d0 = 5.0 m` is NOT calibrated from data** — unlike `tau`, which the P3 sweep chose. It is
   a rig-origin-to-rear-face standstill gap picked to keep the barrier meaningful at v ≈ 0.
   *Unblocked by:* the same sweep machinery, on the low-speed bands where it binds (0–1 m/s: 9 LEAD
   windows in the panel — **UNPOWERED there today**, which is why it was not swept).

---

## 9. FILES

| path | what |
|---|---|
| `stack/tanitad/refs/refav1_lon_cost.py` | the term (new) |
| `stack/tests/test_refav1_lon_cost.py` | 23 tests — arithmetic, integrator parity, one-sidedness |
| `stack/tests/test_refa_v1_dk_hook.py` | 9 tests — planner parity, each paired with a control that MUST differ |
| `stack/tanitad/refs/refa_v1.py` | the hook: `dk_spec` / `lead_gap_m` kwargs, one line in `_cost_chunk`, provenance on the result |
| `taniteval/tools/refav1_arm.py` | `--dk-w/--dk-tau/--dk-d0/--dk-gap-source/--dk-gap-block`, the plan-time gap join with its own speed cross-check, the reached-plan gate, the dump record |
| `…/Research/2026-09-06-refav1-distance-keeping/` | this document, `raw/`, `code/` |

**Suite:** `pytest -q` over `stack/tests` + `taniteval/tests`. The refa_v1/refav1 families alone:
**441 passed, 0 failed** with the hook in place.
