# PREREG · SCORER-FT arm P: does a decision-level term on the planner's own `navsim_v1` pick raise tracked PDMS over the matched control C?

Written 2026-11-24 by the SCORER-FT O3 pre-registration agent (Opus) for the Master Mind. Status: **DRAFT, NOT FROZEN.** The MM freezes it (sha256 in
LOOP_STATE and the register row) **before `weight_rule_rank.py` runs on P's argv and before any P step**. ⛔ Nothing is launched by this document, and the
run needs a PI go (§11). Proposed register key **`H-SCORERFT-PICK-1`**. Proposed repo home: `PK/raw/2026-11-24-scorerft-pick/PREREG_SCORERFT_PICK.md`.
**Amended 2026-11-24, before any P data, for the trainer's gradient clip** (§3.2, CTRL-CLIP, PL3 / PL4; `CARD_QUESTIONS.md` Q-P17).
**Amended again 2026-11-24, before any P data, for MM decision 11:** `train.py` keeps the logging of the pre-clip norm and the clip coefficient
(§3.1 point 3 and CTRL-S0's allow-list, §3.2, §4.2, the pins in §4.10; `CARD_QUESTIONS.md` Q-P18).
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Sources are read at the mirror tip `5eb74f30`.

* **Parent card:** `PK/raw/2026-11-20-scorer-ft/PREREG_SCORER_FT.md` (sha256 `adc1b2ae8d3cc9b3dfed00a20a3c39197428cda8492be34af3bb9ed24010ec91`).
  Its §9 outcome **O3** committed this arm on 2026-11-20, **before any SCORER-FT number existed**: *"decision-level term: pick-anchored pairwise on the
  `navsim_v1` aggregate (every j with PDMS_j > PDMS_pick vs the current pick, weight PDMS_j − PDMS_pick), a new head key `pick` in the same
  `rank_loss.py`; C reused, ~14.2 L40-h"*.
* **Stage 1 read that triggered it:** `PK/raw/2026-11-24-scorerft-read/read_stage1.json` (sha256 `d301d38c6c358a4220a0cdb4cd7641f86c9c4a01d123b546050305e7fda32d41`,
  byte-equal to the dev-box copy `E:/Projects/TanitAD-artifacts/scorerft_read/read_out/read_stage1.json`, MEASURED).
* **Implementation (NOT landed, nothing run on a pod):** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/o3/` (`README.md`, `changes.diff`,
  base `9d11eb0f`): `rank_loss.py` sha256 `3f782697d7c1ec3a392edd7984d3a3060f36a573fd5c20e689f3ddc0e751b71e`, `train.py` sha256
  `fd5e5dadf8c8fbd08658ddf0ef055b7f746d3f4b5a91a8db74f009aa3b8e7ed4` (both LF, MEASURED), profiles `refe_scorerft_p` / `refe_scorerft_p_s1`.
  * ⚠️ **MM decision 10 supersedes part of it** (label-consistent routing, §1.1). That `rank_loss.py` builds the **unrouted** term, so it must be revised
    before the land, together with its gate reference (`gate_checks.rank_reference`, `judge_rank_reach`) and its tests (§4.4).
  * Its sha `3f782697…` and its INHERITED test counts describe the unrouted version only.
  * `train.py` needs no further change, because routing has no flag.
  * **The routed build** is `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/o3b/` (rebased on `5eb74f30`, NOT landed): `rank_loss.py`
    sha256 `d3f0ad40f2ddb51a5b22ef16c90b47f20e7de912686ef4d6f65b841549710291`, `train.py` unchanged at `fd5e5dad…` (MEASURED). Its README §3 records
    that the trainer's global gradient clip bound in a dev-box smoke at `--rank-w` 0.5 and 1.0; §3.2 handles the clip.
  * **The clip-logging build** is `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/o3c/` (rebased on `7ac49a18`, NOT landed). It is o3b
    plus MM decision 11's logging (§3.2) and the CTRL-CLIP probe `PK/refe/clip_probe.py` (sha256 `8fce8590…`). Its `train.py` sha256 is
    `b1f475185237c7518bb86ebcc70f83813951bcf9f317f20f573efb6d14d854b8` (LF), and its `rank_loss.py` is unchanged at `d3f0ad40…` (both MEASURED).
* **The read tool change** this card needs is specified, not implemented: `o3/card/READ_TOOL_SPEC.md`. Open points: `o3/card/CARD_QUESTIONS.md`.

## 1 · Question and hypothesis
**Q.** Does fine-tuning C1 with `L_rank` head **`pick` alone**, trained through the trunk (`--scorer-sees-trunk`), raise **tracked PDMS** over C? C is
the identical fine-tune without the term, and it is **reused** from Stage 1 (§3.1).

The `pick` term is FROZEN as the parent's committed O3 term: pick-anchored pairwise on the `navsim_v1` aggregate, gap-weighted, **with MM decision 10's
label-consistent routing (§1.1)**:
* p_h(x) = sigmoid(logit_h(x)) for h ∈ {NC, DAC, EP, TTC, C}. The aggregate is s(x) = `navsim_v1(p(x))` = p_NC·p_DAC·(5p_EP + 5p_TTC + 2p_C)/12, the
  expression `REFePlanner.aggregate` uses under rule `navsim_v1`.
* k = argmax_x s(x), the scorer's own pick. It is detached and taken in the dtype the logits arrive in (float32, as the planner computes it).
* Pairs: every j whose **label** PDMS_j = `navsim_v1(y_j)` is **strictly** greater than PDMS_k. Each pair has weight w_j = PDMS_j − PDMS_k and loss
  softplus(s̃(k) − s̃(j)).
  * s̃ is s computed on the routed probabilities p̃ of §1.1. Its forward value equals s bit for bit.
  * An unsafe pick has PDMS 0, so it is paired with every proposal of PDMS > 0.
  * Because PDMS_j > 0, every j has DAC label 1 and NC label 0.5 or 1.
* **Gradient reach.** The term reaches the NC, DAC, EP, TTC and C logits (where §1.1 lets it), `score_dec`, `score_q_mlp`, and the trunk (LoRA +
  `scene_proj`). DDC and the trajectory decoder get **exactly 0.0**. The unrouted form of this reach is INHERITED from the implementer's
  `trainer_pick_reach` test and is re-tested on the routed build (§4.4).

### 1.1 Label-consistent routing (MM decision 10, taken 2026-11-24 before any P data exists)
* **The rule.** The term's gradient reaches a sub-score head's probability p_h(x) only in the direction of that proposal's own harness label y_h(x). Per
  factor, inside s:
  * p̃_h(x) = p_h(x) where the factor is **live**, and stopgrad(p_h(x)) otherwise.
  * **On the pick k** (the term lowers s(k)): live ⇔ y_h(k) < p_h(k).
  * **On a better j** (the term raises s(j)): live ⇔ y_h(j) > p_h(j).
* **The mask** is a function of the labels and of the detached p, and carries no gradient.
  * Each proposal of a set has at most one role, because the pick is never a j of its own set (strict inequality). So one mask per (proposal, head) is
    well defined.
  * Every partial ∂s/∂p_h ≥ 0, so a live factor moves only toward its label.
* **What it changes, and what it does not.**
  * The forward value s̃ = s, the pick k, the pairs, the weights and the **loss value** are unchanged bit for bit.
  * Only the gradient is masked.
  * There is **no new flag and no new lever**: routing is part of head `pick`, and it is the parent's committed O3 term.
* **Consequences.** σ is never exactly 0 or 1, so the mask can be read off the labels:

  | label | on the pick | on a better j |
  |---|---|---|
  | y_h = 1 | **never** gets gradient (that head's probability is never lowered) | always live |
  | y_h = 0 | always live | never gets gradient |
  | partial NC 0.5, continuous EP | depends on p | depends on p |

  * A **safe pick's** NC and DAC (labels 1) therefore receive **exactly 0.0** from the term. This removes, by construction, the push against the labels
    identified in `CARD_QUESTIONS.md` Q-P3.
  * An **unsafe pick's** failing gate (label 0) is lowered, as it should be.
* **Why (MEASURED, Stage 1).** T's gate regret was **+1.27** [+0.98, +1.57], and zeroed picks went from **2.81 % to 4.07 %**. The unrouted `pick` term
  would push a safe pick's NC / DAC against their labels.
* **Scope, stated honestly.** The guarantee is at the level of **each proposal's own logit**: the term never moves a head output away from that
  output's label.
  * It is **not** a guarantee at the parameter level. `score_head`, `score_dec` and the trunk are shared across proposals and tokens, so a live gradient
    on one proposal can still move another proposal's output.
  * That residual path is what M2 / Zup (§7) and G1 watch.
* **Controls.** CTRL-ROUTE (§4.4) checks it analytically, checks it on the real smoke, and has a RED mutation.

**H-SCORERFT-PICK-1 (HYPOTHESIS).** On full navtest, P − C tracked PDMS > 0. The effect replicates at a second training seed and lies above that seed
pair's own floor. No NC / DAC / EP / TTC / C sub-score is separated worse.

**Why this arm (from Stage 1, MEASURED).** T's head-level term (pairs `ep` + `gate`) worked on the heads:
* EP Spearman rose by **+0.353** [0.334, 0.375];
* the EP share of the regret fell by **−0.91** [−1.00, −0.82].

The decision paid for it at the gates:
* regret_gate **+1.27** [+0.98, +1.57];
* regret_ttc **+0.52** [+0.33, +0.72];
* zeroed picks rose from 2.81 % to 4.07 %;
* P1 = **−0.9015** [−1.2021, −0.5874].

Each head was scored on its own ranking, while the planner ranks their **product**. The `pick` term scores the product the planner argmaxes against the
label PDMS the harness would assign. A trade such as EP for a gate, which loses PDMS, therefore carries a loss of its own.

## 2 · Basis (every number: class, artifact, tier, estimator)
**MEASURED.** Source: `PK/raw/2026-11-24-scorerft-read/read_stage1.json`.
* Tier: T1, NAVSIM v1 tracked PDMS ×100.
* Population: 12,146 tokens over 136 logs.
* Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927.
* One training seed per arm.

| quantity | C (the reused control) | T − C |
|---|---|---|
| tracked PDMS | **92.67** [91.90, 93.40] | **−0.9015** [−1.2021, −0.5874] |
| oracle-of-64 | 99.34 [99.19, 99.48] | +0.0304 [−0.0311, +0.0901] |
| regret (oracle − pick) | **6.67** [6.02, 7.37] | +0.9320 [+0.6091, +1.2408] |
| regret: gate / EP / TTC / C / other | 2.58 / 2.61 / 1.44 / 0.001 / 0.05 | +1.27 / −0.91 / +0.52 / +0.001 / +0.05 |
| top-1 rate (pick = oracle) | 48.0 % | +1.60 pp [+0.44, +2.80] |
| zeroed-pick rate | 2.81 % | +1.26 pp [+0.97, +1.55] |
| EP Spearman, safe proposals | 0.401 | +0.353 [0.334, 0.375] |
| pick confidently wrong at the boundary: DAC / NC | 67.9 % of 193 / 57.8 % of 128 | (reported) |

* Every Stage 1 read control passed, including CTRL-DET (the full seam of T re-run bit-identically) and CTRL-W0 (C's step-0 digest = C1's `f09949e2…`).
* **Step cost of C, MEASURED.** C's pod config gives `started_at` 2026-11-22T23:01:48Z. C's summary gives `finished_at` 2026-11-23T13:14:18Z.
  * That is 51,150 s for 1,000 steps, **51.2 s/step ⇒ 14.2 L40-h** on an NVIDIA L40 with torch 2.7.1+cu128.
  * Sources: `pod_config_c.json` sha256 `a66f6750…`, `pod_summary_c.json` sha256 `ad662edf…`.
  * UNVERIFIED that no restart split the run: the argv carries `--resume`.
* **Step 0, MEASURED on the banked run logs** (§3.1, CTRL-S0's demonstration). C's and T's step-0 training rows agree exactly on all 19 compared
  keys (`traj_L1` 0.2121780663728714 in both), and their step-0 `onpolicy` rows agree on all 10.
* **INHERITED, not re-run here.** These results are for the **unrouted** build and must be re-run on the routed one (§4.4). Source: the implementer's
  `o3/README.md` and `o3/tests.tsv`.
  * `pick_identity`: `RK.navsim_v1` equals `REFePlanner.aggregate` bit for bit, and the pick index equals the planner's argmax.
  * `trainer_pick_reach`. On the routed build (o3b) it compares `--rank-w` 0.001 with 0.002 (exact 2×), because 0.5 and 1.0 both sat on the
    trainer's clip in its tiny synthetic smoke (ratio 1.065; o3b README §3, INHERITED). The clip makes only a ratio between two weights fail; §3.2.
  * Test counts: `test_rank_loss.py` 255 (46 RED mutations), `test_scorerft_profile.py` 415, `test_launch_gate.py --static --no-banked` 1180.
* **Register context, all on C1 and all on SNOOPED populations.** None of these numbers is imported as a bar here. Together they say *train the
  decision*, not *re-weight frozen outputs*.
  * `D-REFE-SRULE-1`: no fixed reweighting of the heads beats the pick, −0.17 [−0.32, −0.01].
  * `D-REFE-SRERANK-1`: learned re-rankers on frozen outputs find NO SIGNAL.
  * `D-REFE-HEADATTR-1`: the regret has three terms of similar size, and the EP head is the weakest.
* **ESTIMATED:** the effect size. No prior result predicts it; the power is in §5.4.

## 3 · Arms (same init, same bank, same steps; the only differences are listed)
| arm | profile | status | seed | argv vs C |
|---|---|---|---|---|
| **P** | `refe_scorerft_p` | to train | 0 | + `--rank-w w*_P` + `--rank-heads pick` + own `--out` |
| **C** | `refe_scorerft_c` | **REUSED** (Stage 1: code `code_83411c28`, `model_final` sha256 `d40dd516…`, HF `Sayood/tanitad-refe-p5c` prefix `scorerft/c`) | 0 | C1's recipe + `--init-from <C1 final>` + `--steps 1000` + `--rank-w 0.0` |
| **Ps** | `refe_scorerft_p_s1` | Stage 2, conditional | 1 | P + `--seed 1` |
| **Cs** | `refe_scorerft_c_s1` | Stage 2, conditional (the parent's Cs profile, never run) | 1 | C + `--seed 1` |

* **The argv identities.**
  * Recipe = C's banked argv exactly (`PK/raw/2026-11-24-scorerft-read/argv_c.json`, sha256 `00729d56…`): init C1 final `f09949e2…`, 1,000 steps,
    cosine to 0, lr 5e-5, batch 4 × accum 64, fresh AdamW, bank `train_grow_cmd_3c_p7` (parity, no re-selection), label table `3b198314…`,
    `--scorer-sees-trunk`.
  * **argv(P) − argv(C) = {`--rank-w`, `--rank-heads pick`, `--out`}**, exactly.
  * argv(Ps) − argv(P) = argv(Cs) − argv(C) = {`--seed 1`, `--out`}.
  * P names neither `--rank-ep-gap` nor `--rank-gate-topk`: the trainer refuses them without `ep` / `gate`, and `pick` never reads their defaults
    (0.02 / 8).
* **Normalisation (MM decision 1, frozen).**
  * Per set: L(b) = Σ_j w_j softplus(s̃(k) − s̃(j)) / Σ_j w_j. Its value is unchanged by the routing of §1.1.
  * A set with no pair (k is already the label-best) contributes exactly 0 **and is counted**.
  * L_rank = (1/`--batch`) Σ_b msk_b L(b). This is the same constant denominator as `ep` / `gate` / `ttc`, so `--accum` stays exactly one large batch.
* **No temperature and no new lever (MM decision 2, frozen).** w*_P comes from `weight_rule_rank.py` run on **P's own argv**, on the pod, at C1's init,
  with real `pdm` labels. The rule is the parent's: the smallest w ∈ {1/64, …, 4} with r(w) ≥ 1.0.
  * **K2** (r(4) < 1, or `pick` in `dead_heads`) **or K2b** (ρ(w*) > 1) ⇒ **P is not launched** (§9, PL1 / PL2).
  * `SCORERFT_P_RANK_W` stays `None` until the rule returns SELECT. The gate may only veto w*, never retune it.
  * **The gradient clip does not reopen the rule** (§3.2). r and ρ are ratios inside one gradient, and the clip scales the whole gradient by one
    coefficient, so both are exactly clip-invariant. What the clip does to P is audited by CTRL-CLIP (§3.2), not by changing the rule.
* **Staging.**
  * **Stage 1** = P alone, with C reused.
  * **Stage 2** = Cs, then Ps, run **only if P1 and G1 hold** (MM decision 4).
* **Hardware.** P runs on the same pod class and software as C: NVIDIA L40, torch 2.7.1+cu128 (C's `pod_config_c.json`). CTRL-CODE reads this.
* **T and Ts are RETIRED** (Stage 1 read O3, so no Stage 2). Their profiles stay registered but are not run by this card (§13).

### 3.1 Why C may be reused (MM decision 3): the source argument, then the controls that test it on the run
Static. MEASURED from source at `5eb74f30` and from the o3 diff:
1. **The term is never built for C.** C ran `--rank-w 0.0`, so `rank = None` (`PK/refe/train.py:1591`).
   * Every use of `rank` sits behind `rank is not None` or `l_rank is not None` (`train.py:1592, 1804, 1863, 1996, 2008, 2074, 2105`), and so does
     every `RK.*` call but one.
   * The one exception is `RK.parse_heads(a.rank_heads)` inside `rank_refusals` (`train.py:1120`). With `""` it returns `()` before it reads `HEADS`.
2. **The `rank_loss.py` change** is 28 changed lines against `5eb74f30` in the unrouted o3 build:
   * the docstring;
   * the `HEADS` and `MUTATIONS` tuples;
   * a new `pick_index`;
   * a branch under `if "pick" in self.heads:` inside `RankLoss.pairs`.

   None of them is reachable when `RankLoss` is not constructed. The routing of decision 10 adds only lines inside that same `pick` branch, plus a
   `pick_noroute` entry in `MUTATIONS`. The code audit (§4.2) checks this on the landed diff.
3. **The `train.py` change** is two **string-only** hunks, a refusal message and an argparse help text (MEASURED by diffing the o3 copy against
   `5eb74f30`), plus the **read-only clip logging** of MM decision 11.
   * The logging binds the norm that the unchanged `clip_grad_norm_` call already returns, and writes two new keys into the logged row (§3.2; hunks
     listed in §4.2).
   * The parameters it produces are bit-identical to those produced without it (MEASURED on CPU, §3.2). So the point still holds: on C's argv, P's
     `train.py` computes what C's computed.
4. **C's code tree and the tip match outside the launch gate.**
   * Between C's code tree `83411c28` and `5eb74f30`, the only files under `PK/refe/` that differ are `gate_checks.py`, `launch_gate.py` and
     `test_scorerft_profile.py`, and `train.py` imports none of them. (`git diff --stat 83411c28 5eb74f30 -- PK/refe/`)
   * `train.py` at `83411c28` hashes to `59a737f4…`, which equals the `source_sha256["train.py"]` recorded in **both** C's and T's pod configs.
5. **P adds no model parameter**; the term is a loss only. The checkpoint format and the navtest harness therefore load P exactly as they loaded C.

⇒ **On C's argv, P's code computes what C's code computed.** That is a claim about code. The run-time controls test it:
* **CTRL-S0, FULL form** (MM decision 3; the only form). P's step-0 logged metrics must equal C's **exactly**.
  * **C's side** is the banked run log `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/o3/card/evidence/metrics_scorerft_c.jsonl` (3,046
    lines, sha256 `97a8631cc863d95cf646932ef3833206affbc7a0857d21bff5d5e8b78ff04eed`, pulled framed and equal to the pod's per the MM).
  * **P's side** is P's own `metrics.jsonl`, pulled and banked the same way.
  * **The rule.** For a step, the **training row** is the last row with `step == 0` and no `event` key, and the **`onpolicy` row** is the row with
    `event == "onpolicy"`.
    * Every key C's row carries, except `elapsed_s`, `cuda_max_mem_gb` and `at` (wall-clock / allocator), must be present in P's row and `==` equal.
    * A key only P carries must match `rank_L | rank_w | rank_pairs_* | rank_sets_* | pick_v1 | random_v1 | best_v1 | skill_v1 | grad_norm_pre |
      clip_coef`. The last two are MM decision 11's logging keys (§3.2). C's banked log predates them, so they are P-only keys, exactly as T's rank
      keys were.
    * P's `rank_L` is finite, and its `rank_w` equals its argv.
  * **The exact keys compared:**
    * training row (19): `assign_d, batch, bce_c, bce_dac, bce_ddc, bce_ep, bce_nc, bce_ttc, epoch, lr, pos, pos_L1, samples, score, scorer_cov, step,
      traj_L1, winners, yaw_L1`;
    * `onpolicy` row (10): `best, cov_norm, drift_m, event, lag_steps, pick, random, sets, skill, step`.
  * **Demonstration on Stage 1, MEASURED 2026-11-24.** T (`evidence/metrics_scorerft_t.jsonl`, 3,047 lines, sha256 `09cdaf83c611fba9af7bafef250affc7195af71aa9c40bfcf461888d712b0f6a`)
    against C, under exactly this rule:
    * **19 / 19 training keys equal and 10 / 10 `onpolicy` keys equal.** Examples:

      | key | value in both |
      |---|---|
      | `traj_L1` | 0.2121780663728714 |
      | `score` | 0.014537446200847626 |
      | `bce_nc` | 0.17235779762268066 |
      | `pick` | 0.950812254101038 |
      | `lr` | 4.999987663004646e-05 |
      | `pos` / `samples` | 256 / 256 |

    * T-only keys: `rank_L, rank_pairs_ep, rank_pairs_gate, rank_sets_ep, rank_sets_gate, rank_w`, and `best_v1, pick_v1, random_v1, skill_v1`. All
      are in the allowed set. Excluded keys present: `elapsed_s` (62.84 vs 63.15 s) and `cuda_max_mem_gb`.
    * ⇒ **CTRL-S0 reads PASS on T vs C.**
    * **The control is not vacuous.** The same rule applied to **step 1** FAILS on 10 training keys (`bce_c, bce_dac, bce_ddc, bce_ep, bce_nc,
      bce_ttc, pos_L1, score, traj_L1, yaw_L1`) and on 4 `onpolicy` keys (`best, pick, random, skill`), because T's first update differs from C's.
* **CTRL-ORDER:** the per-step (epoch, pos, samples, batch, lr) series of P equals C's over all 1,000 steps. These are pure functions of the loader
  and the schedule, so they prove the data stream is the same after step 0.
  * **Demonstration on Stage 1, MEASURED:** T vs C, 1,000 / 1,000 training rows in each log, no `resume` event in either, **0 of 1,000 steps differ**.
* **CTRL-W0** (kept): C's step-0 digest **and** P's both equal C1's sha256
  `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4`.
* **CTRL-CODE:** P's and C's pod `config.json` must agree on:
  * the same `device` and `torch`;
  * the same `source_sha256` for `model.py`, `load_dinov3.py` and `ckpt_io.py`;
  * a `train.py` source sha equal to the landed P blob;
  * the same `warm_start.sha256`, `targets_sha256`, `traj_label_sha256`, `calib_sha256` and `frozen_sha256`;
  * the same parsed `args`, except `rank_w`, `rank_heads` and `out`. This covers the defaults that argv cannot show.
* **CTRL-HARNESS:** P's seam was made by the same harness export as C's (`751f3ae7`, `sft_seam_chain.sh`). Its seam report carries the same
  `select / rule / nav_mode / goal_fix / repair_last_heading / sanitize_goal`.
* **CTRL-CREUSE:** the C the read uses **is** Stage 1's C.
  * Its seam and dump sha256 equal Stage 1's `seam_ledger.json`, and its harness CSV sha256 equals Stage 1's.
  * Its levels (seven metrics × {tracked, transparent, TD} × {point, lo, hi}) and its selection-quality levels equal `read_stage1.json`'s to 1e-9. The
    draw matrix depends only on the logs, the draw count and the seed, so the CIs must also reproduce.

⛔ **If CTRL-S0, CTRL-ORDER or CTRL-CODE fails, C is not reusable.** The read is INVALID and prints no outcome. The committed response is
**C′ = C re-run on P's code and pod** (~14.2 L40-h, a new PI go), **never a tolerance chosen after the fact**.

### 3.2 The gradient clip (Q-P17; decided 2026-11-24, before any P data)
**The facts.** Each is MEASURED unless marked otherwise.
* **The clip.** The trainer clips the global gradient norm at a fixed **1.0** over every trainable parameter (`train.py:2040`, the same line at
  `5eb74f30` and in o3b's `fd5e5dad…`). There it **discards** the pre-clip norm N_t that `clip_grad_norm_` returns. o3c's `train.py` binds that norm
  and logs it, by MM decision 11 (see **Scope** below).
* **No banked log carries a norm.** Not one of the rows of `evidence/metrics_scorerft_c.jsonl` (3,046) or `_t.jsonl` (3,047) has a gradient-norm key
  (every key enumerated). ⇒ **Whether the clip bound in C or in T was never measured, and no banked artifact can recover it.**
* **The weight rule never reaches the clip.** `weight_rule_rank.py` (sha256 `a28e3ee7…`, identical at `7ac49a18` and `5eb74f30`) stops at the
  optimiser build. It takes each term's gradient separately with `torch.autograd.grad`, so it is pre-clip by construction. What it does and does not
  measure:
  * only l_score on S, L_rank on S and the trunk, and l_traj on the trunk;
  * no total and no global norm;
  * a pool of 8 micro-batches, where the trainer's step pools 64.
* **T's registered output cannot show the clip binding at T's w\* = 1/32** (`PK/raw/2026-11-22-scorerft-rankw/weight_rule_rank.json`, sha256
  `a9f13c78e987440c8255bff4910e2814c46bc85c0459caa7825cf99d745dc401`).
  * At w\*, the four norms it measured are 0.0992 (l_score, S), 0.1736 (L_rank, S), 0.7132 (l_traj, trunk) and 0.2086 (L_rank, trunk).
  * Their sum is 1.195 and their root-sum-square is 0.770, so they straddle 1.0.
  * l_score on the trunk and l_traj on the trajectory decoder were never measured. ⇒ **UNDETERMINED.**
* **The gate smoke cannot answer it either.**
  * The smoke steps with `--accum` 2 (o3b `launch_gate.py:246, 2086-2087`), while the trainer steps with 64. A window's norm falls with the number of
    micro-batches it pools: in T's output, the per-micro-batch RMS norm is **1.6–3.0×** the 8-pooled norm for each of the four gradients.
  * So a smoke step can sit on the clip where a real step does not. The o3b smoke that put `--rank-w` 0.5 and 1.0 on the clip also used a dev-box
    tiny bank, synthetic labels and a zeroed BCE (INHERITED).

**What the clip can and cannot change.** This is analytic, with a toy check. `clip_grad_norm_` multiplies **every** gradient by one coefficient,
c_t = min(1, 1/(N_t + 1e-6)) (torch 2.7.1, `_clip_grads_with_norm_`). Three consequences follow.
1. **r(w) and ρ(w) are exactly clip-invariant.** Each is a ratio of two terms' norms inside one window, and c_t cancels.
   * ⇒ They mean for P what they meant for T. **The weight rule, its candidates, K2 and K2b are unchanged.**
2. **AdamW is invariant to a constant c.** In m̂/(√v̂ + ε), c cancels except through ε. ε is the default 1e-8, because `train.py:1541` passes none, and
   the decoupled decay never reads the gradient.
   * ⇒ A clip that binds by the same factor on every step leaves P's training as it was.
3. **Only a coefficient that varies across steps acts.** It reweights steps in Adam's moments, and that includes l_traj's and BCE's steps whenever
   the term changes which steps are clipped.
   * **Toy, MEASURED.** Source: `evidence/adamw_clip_toy.py` (sha256 `5f00c786…`) and its output `.txt` (`f4ad79ae…`). CPU, float64, torch 2.7.1,
     the trainer's AdamW at lr 5e-5 and wd 0.01, 200 steps.
   * The end-point difference from the unclipped run, as a share of the unclipped run's movement:

     | rescale | share |
     |---|---|
     | constant ×0.3 | **5.3e-5** |
     | alternating ×1 / ×0.3 | **0.125** |
     | the real clip, binding on every step with a varying N_t | **0.091** |

⇒ **The rule is not re-derived.** What must be measured is **whether the term changes which steps the clip acts on**, against C.

**CTRL-CLIP: a pre-launch probe in two forms, C-form and P-form.** It is an instrument that keeps no weights.
* **When and where.** It runs on P's landed code, on the L40 pod with C's image, after P's launch-gate PASS token (§4.4) and before P's launch.
* **Argv.**
  * C-form: C's banked argv `argv_c.json` (`00729d56…`).
  * P-form: P's PASS-token argv.
  * In each, only `--out` is replaced, by a scratch directory. `--steps 1000` is kept, so the lr on the probed steps is the real run's.
* **Mechanism.** The probe runs the real `train.run(a)`, the build technique of `weight_rule_rank.py`, with a pass-through wrapper on
  `torch.nn.utils.clip_grad_norm_`.
  * The wrapper calls the original and returns its value **unchanged**.
  * It records N_t, and c_t computed with torch's own expression in N_t's dtype.
  * It stops before the optimiser step of step 8. It writes no checkpoint, because C's first `ckpt` event is at step 24 (MEASURED).
  * Its scratch directory is deleted once the JSON is banked.
* **K = 8 steps (steps 0–7).** 8 is `N_MICRO`, the registered size of the weight rule's own init measurement, so the probe adds no number and keeps the
  rule's scope, the start of training.
  * Cost: C's steps 0–7 took **417.5 s** (MEASURED, `elapsed_s` at step 7), so the two forms take ~0.23 L40-h plus two builds.
* **Binding.** b_t = [c_t < 1.0]: exactly "torch scaled step t down". No tolerance.
* **Validity.** Any FAIL ⇒ PL3 (§9).
  * **V1, dev-box unit test:**
    * a known norm of 2.0 records b = 1, and a known norm of 0.5 records b = 0;
    * the wrapped clip leaves the gradients bit-identical to the bare clip;
    * RED: the mutation "record the norm after clipping" must fail it.
  * **V2:** each form's step-0 training and `onpolicy` rows pass CTRL-S0's rule (§3.1) against C's banked step-0 rows. For the P-form this is also an
    early read of CTRL-S0; it does not replace CTRL-S0 on P's run.
  * **V3:** each form's (epoch, pos, samples, batch, lr) for steps 0–7 equals C's banked rows.
  * **V4:** 8 finite N_t per form, and no `resume` event. In each form, the trainer's own logged `grad_norm_pre` / `clip_coef` for steps 0–7 must
    equal the wrapper's N_t / c_t **exactly** (MM decision 11; `clip_probe.py` V4).
* **The read.** Steps are paired by index; by V3 they share samples and lr.
  * n_on = #{t : b_t^P = 1, b_t^C = 0}: steps the term puts on the clip.
  * n_off = #{t : b_t^P = 0, b_t^C = 1}: steps the term takes off it.
  * **CLIP-SAME** ⇔ n_on = n_off = 0: the clip acts on the same steps in both arms. It is labelled **CLIP-0** when no step of either form binds.
  * **CLIP-T** ⇔ n_on + n_off > 0: the term changes the steps the clip acts on.
  * Reported, deciding nothing: occ_P = Σb^P/8, occ_C = Σb^C/8, the minimum c_t of each form, and N_0^P / N_0^C.
  * At step 0 the two forms share weights, samples and labels, so |N_0^P − N_0^C| ≤ ‖w\*·∇L_rank‖ ≤ N_0^P + N_0^C. That is the first global measure
    of the term's size, bounded on both sides.
* **Why CLIP-T does not veto the launch** (PL4, §9).
  * The clip is a recipe constant that C and P share, so P − C stays the causal effect of adding the term to that recipe.
  * The rule's ratios are exact under the clip (point 1).
  * No lever-free alternative exists. A smaller w breaks r ≥ 1, and a different clip threshold is a new lever that needs C′.
  * What CLIP-T changes is the **scope of the claims**.
* **Scope.** The probe alone decides PL3 / PL4, and it covers steps 0–7 only.
  * **MM decision 11 (Q-P18): P's `train.py` logs N_t and c_t on every logged training row,** as `grad_norm_pre` and `clip_coef`. c_t is computed
    as clamp(1.0 / (N_t + 1e-6), max = 1.0) in N_t's dtype, torch's own expression. The hunks are listed in §4.2. The logging is behaviour-neutral,
    so it does not weaken the reuse of C:
    * **Bit identity: MEASURED by the o3c implementer, INHERITED here** (the card author read the test source and did not re-run it).
      * Test: `test_clip_probe.py`, `chk_logging_changes_no_behaviour`; `tests.tsv` reads PASS, `ZZCLIPPROBE_PASS 52`.
      * On the CPU trainer, after 3 steps, every final checkpoint tensor is `torch.equal` with and without the logging. The no-logging copy is
        `train.py` with the capture and both keys removed by source mutation.
      * **Determinism control:** current vs current is bit-identical.
      * **Sensitivity arm:** a copy that skips `opt.step()` DIFFERS, so the comparison can fail.
    * **Source: MEASURED by the card author.**
      * `gn_pre` occurs on exactly two lines: `train.py:2040`, where it is bound, and `:2065`, where it is read into the row.
      * The `clip_grad_norm_` call, its parameter list and its max_norm 1.0 are unchanged.
      * The row is built after `opt.step()`, with no in-place operation and no RNG draw.
      * On the L40, bit identity is argued from this source and not measured. CTRL-S0, CTRL-ORDER and CTRL-CODE still run on P's own run (§3.1).
    * **The step-0 allow-list.** The two keys are P-only keys in CTRL-S0's rule (§3.1; `S0_RANK_ONLY`, o3c `scorerft_read.py:108`). C's banked log
      predates them, so they are allowed exactly as T's rank keys were. The 19 + 10 keys compared on C's rows are unchanged. The C-form probe runs
      on P's landed `train.py`, so its rows carry the two keys too, and V2 allows them on the same rule.
    * **What the logging adds:**
      * the V4 cross-check above;
      * P's own full-run N_t / c_t, banked in P's `metrics.jsonl`. They are descriptive and decide nothing: no bar or outcome reads them.
    * **C's full-run clip path stays UNMEASURED,** because C's banked log predates the logging and C is not re-run. So Stage 1 has no P-vs-C
      clip comparison beyond steps 0–7.
  * ESTIMATED: steps 0–7 are expected to bound the run's occupancy from above, because the term weakens as the pick improves (Q-P6) and l_traj falls.
    That can now be checked on P's side only, and only as a description.
* **CTRL-ROUTE and the reach checks are clip-invariant.** A coefficient c_t ∈ (0, 1] keeps an exact 0.0 at 0.0 and a non-zero value non-zero. So the
  only check the clip can break is a ratio between two weights, which is why o3b's reach test uses 0.001 vs 0.002.

## 4 · Admissibility before minute 0
1. **The card is frozen** (sha256 in LOOP_STATE and the register row) before the weight rule runs on P's argv.
2. **The P implementation is landed** on `5eb74f30` or later (the 1c hunks of `gate_checks.py` / `test_scorerft_profile.py` rebased), and the full
   suite is green (G-SUITE, 0 regressions).
   * **Code audit, banked as text:** `git diff 83411c28 <P code commit> -- PK/refe/` may differ, among the files `train.py` imports transitively,
     **only** in `rank_loss.py` and `train.py`. For `rank_loss.py` that means the o3 hunks plus the routing (§1.1), all inside the `pick` branch, the
     docstring and the `HEADS` / `MUTATIONS` tuples. For `train.py`, exactly the four hunks below and nothing else. Otherwise STOP and return the
     card to the MM.

     | `train.py` lines (o3c copy) | hunk | kind |
     |---|---|---|
     | `:1133` | the `--rank-heads` refusal message names `pick` | string (o3) |
     | `:2040` | `gn_pre = ` binds the return value of the unchanged `clip_grad_norm_([…] + aux_params, 1.0)` call | **non-string, read-only (decision 11)** |
     | `:2062–2065` | the logged training row: at 2062 the closing `}` becomes `,`; 2063–2064 are two comment lines; 2065 adds `"grad_norm_pre"` and `"clip_coef"` | **non-string, read-only (decision 11)** |
     | `:2355–2356` | the `--rank-heads` help text names `pick` | string (o3) |

     * The line numbers are those of o3c's `files/train.py` (sha256 `b1f47518…`), in `diffs/train.py.diff` against `7ac49a18`.
     * They are also the audit's hunks against `83411c28`. MEASURED: `train.py`'s blob is the same at `83411c28`, `5eb74f30` and `7ac49a18` (git
       `29ee68e5…`, sha256 `59a737f4…`, LF), and `git diff --stat 5eb74f30 7ac49a18 -- PK/refe/` is empty.
     * The audit also checks that `gn_pre` occurs on exactly two lines of the landed file, the bind and the row (2040 and 2065 in o3c). If a later
       rebase moves the lines, the hunks are matched by content.
     * **`PICK_TRAIN_SHA256` is filled from the LANDED blob, never from the hold copy** (§4.10). o3c's `b1f47518…` is the value it is expected to
       take.
3. **Weight rule.** `weight_rule_rank.py` is run on P's argv on the pod.
   * Required result: `rule.status == "SELECT"`, `heads == ["pick"]`, `dead_heads == []`, `label_backend == "pdm"`, `NOT_A_DECISION == false`,
     `init_from` = C1's final.
   * The JSON is banked with its sha256, and `SCORERFT_P_RANK_W` is filled from `rule.rank_w`.
   * The per-micro-batch r(1) spread and the pair counts are reported.
4. **A launch-gate PASS token** for `refe_scorerft_p`, bound to commit, argv sha256 and data manifests. It needs:
   * G-DVB `rank_*` rows;
   * G-LIVE: `rank_L` finite on every row; L3 reach (NC, DAC, EP, TTC and C trained; DDC and the trajectory decoder exactly 0.0); the L4 `pick` audit
     against the gate's own numpy reference;
   * every deliberate-regression arm turning the gate RED, including `REFE_RANK_MUTATE=pick_oracle` and `pick_noroute`;
   * a smoke with zero `pick` pairs is REFUSED;
   * **CTRL-ROUTE** (MM decision 10), in two forms. Both read the gradient of `L_rank` **alone** on the score **logits** (BCE and l_traj removed, as in
     L3).

     **(a) Analytic, `test_rank_loss.py`.** On constructed sets:
     * on a pick whose **NC label is 1**, ∂L_rank/∂logit_NC(pick) is **EXACTLY 0.0**;
     * the same holds for every head with label 1 on the pick, and for every head with label 0 on a better j;
     * **non-zero controls:** a pick with DAC label 0 gets a non-zero ∂/∂logit_DAC, whose sign lowers it; a j with DAC label 1 gets a non-zero one
       that raises it; a pick with p_EP > y_EP gets a non-zero EP gradient;
     * **forward identity:** the loss value, the pick index and s are bit-identical to the unrouted computation.

     **(b) Real data, G-LIVE L3 on the smoke's labelled sets.**
     * On every set whose pick has NC label 1, the pick's NC-logit gradient is exactly 0.0. The same holds for DAC.
     * The number of such sets is reported and must be > 0, or the check is vacuous and REFUSED.
     * At least one live cell is non-zero.

     **RED:** `REFE_RANK_MUTATE=pick_noroute` (the routing removed, so s̃ = s with every factor live), and the source mutation that deletes the
     stop-gradient line, must each make (a) and (b) read non-zero, so the gate goes RED.
     * The guarantee is **logit-level**. The `score_head` NC weight row still receives gradient through other proposals' live NC factors. A zero there
       is neither expected nor claimed (§1.1, scope).
5. **P runs on the L40 pod**, with the same torch 2.7.1+cu128 build and image as C. Another GPU class ⇒ CTRL-S0's exact form cannot hold ⇒ C′ is needed
   first (new PI go).
6. **C's run log is BANKED** (MM, 2026-11-24): `o3/card/evidence/metrics_scorerft_c.jsonl`, 3,046 lines, sha256 `97a8631c…`. T's is beside it
   (`metrics_scorerft_t.jsonl`, sha256 `09cdaf83…`) as CTRL-S0's and CTRL-ORDER's Stage 1 positive control.
   * CTRL-S0 runs in its **full form only**; there is no fallback.
   * P's `metrics.jsonl` is pulled framed and banked with its sha256 before the read.
7. **After training,** the post-training G-ADAM gate (`p5c_postgate.py` / zerograd) passes on P: Adam state exists on every trainable parameter.
   * The reach evidence for `pick` is G-LIVE L3, where the BCE term is zeroed. Adam state on `score_head` is not reach evidence here, because the BCE
     term trains those rows anyway.
8. **C's banked artifacts are unchanged**, checked by sha256 before the read:

   | artifact | sha256 |
   |---|---|
   | seam | `914bdd0a…` |
   | dump | `848e6561…` |
   | harness CSV | `2df7f856…` |
   | argv | `00729d56…` |
   | `w0_c.json` | `2dd2dcd2…` |
   | `pod_config_c.json` | `a66f6750…` |
   | `read_stage1.json` | `d301d38c…` |
   | `seam_ledger.json` | `e627df2a…` |
9. **CTRL-CLIP** (§3.2) runs after item 4's PASS token and before P's launch, on the pod of item 5.
   * Its JSON is banked with its sha256 (pin `CLIP_PROBE_SHA256`, filled by the MM), together with both forms' scratch `metrics.jsonl`.
   * PL3 and PL4 (§9) apply.
   * The probe is an instrument run, like `weight_rule_rank.py`: it keeps no weights and runs only gate-validated argvs (Q-P17 asks the MM to confirm
     this reading of the launch-gate rule).
10. **The pins** (Q-P13, settled with MM decision 11, Q-P18). They are constants in `PK/eval/scorerft/scorerft_read.py` (o3c lines 82–85), and all
    are `None` in the hold. The MM fills each one, never an agent, and records each value in LOOP_STATE beside the card's sha256.

    | pin | value | filled when | check |
    |---|---|---|---|
    | `PICK_PREREG_SHA256` | sha256 of this card's frozen bytes, as landed at `PK/raw/2026-11-24-scorerft-pick/PREREG_SCORERFT_PICK.md` | in the **freeze-and-land commit**, which lands the frozen card and P's implementation together (items 1 and 2) | equals the sha256 in LOOP_STATE and in the register row; after the land, the landed blob's sha256 equals the pin |
    | `PICK_TRAIN_SHA256` | sha256 of the **landed** `train.py` blob, `git cat-file blob <land>:"<PK>/refe/train.py" \| sha256sum`. Quote the path, because it contains `&` | in the **freeze-and-land commit**: computed from the staged blob, then re-read from the landed commit | expected `b1f47518…` (o3c, LF). A landed blob that differs, for example after a lander CRLF rewrite or a rebase hunk, ⇒ STOP: re-run item 2's audit on the landed diff, then pin the landed value. It must also equal P's pod `config.json` `source_sha256["train.py"]` (CTRL-CODE) |
    | `CLIP_PROBE_SHA256` | sha256 of the banked CTRL-CLIP probe JSON (item 9) | **not** in the freeze-and-land commit (see below). Recorded in LOOP_STATE **before P's launch**, as soon as the JSON is banked; the constant lands in a later commit that touches only `scorerft_read.py`, before the read | the constant equals the LOOP_STATE record, and CTRL-CLIP (§5.1) checks the probe JSON against it |

    * **Why `CLIP_PROBE_SHA256` cannot be filled in the freeze-and-land commit.** The probe runs on that commit's code, after the PASS token that
      binds it (items 4 and 9). The binding act is the LOOP_STATE record made before P's launch: the probe cannot be swapped after P data exist.
    * **Effect on the code audit.** `scorerft_read.py` sits outside `train.py`'s imports, so neither pin commit moves item 2's audit.
    * ⛔ **While any of the three is `None`, a real read yields no outcome.** MEASURED from o3c's source:
      * `PICK_PREREG_SHA256` or `PICK_TRAIN_SHA256` unset ⇒ the read **REFUSES** before it reads any input (`scorerft_read.py:1198–1201`, raised at
        `:1226–1227`).
      * `CLIP_PROBE_SHA256` unset ⇒ CTRL-CLIP FAILS (`:968–969`) ⇒ **INVALID, no outcome** (`:1023–1024`).
      * Dry runs are not real reads: the tests monkeypatch the pins.
    * `PICK_CODE_TAG` and `SCORERFT_P_RANK_W` keep the Q-P13 timing (at launch, and from the rule JSON respectively).

## 5 · Primary metric, estimator, bar (MM decision 4: identical to the parent)
**Metric:** tracked PDMS ×100 (NAVSIM v1.1, LQR-tracked; **tier T1**) on full navtest, 12,146 tokens over 136 logs.

**Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, **ONE draw matrix** for every arm and contrast. The question
each source of variance gets:
* **Another draw of LOGS:** this is the only question the bootstrap answers.
* **Another TRAINING run:** answered by the seed pair (§8) and by nothing else.
* **Another INFERENCE run:** the planner is a deterministic argmax. Inference variance was measured at 0.0 on C1 and on T (CTRL-DET, full seam), and is
  re-checked on P (CTRL-DET).

| clause | definition | pass |
|---|---|---|
| **P1** | Δ₀ = P − C | **CI-lo > 0** |
| **G1** | each of NC, DAC, EP, TTC, C (×100), P − C | **none separated adverse** (CI-hi < 0) |
| **P2** | Δ₁ = Ps − Cs, and G1 on Ps − Cs | CI-lo > 0, none adverse |
| **P3** | F = \|Cs − C\| (this rig's control floor; Cs is the parent's never-run Cs); d = (Δ₀ + Δ₁)/2 | **d > 2F** (strict; F is the point \|mean(Cs) − mean(C)\|; s_p = \|Δ₀ − Δ₁\|/√2 reported beside it, deciding nothing) |
| **ADOPT** | P1 ∧ G1 ∧ P2 ∧ P3 | as a scorer-training lever on the C1 recipe |

* DDC is reported, not a guard (`navsim_v1` ignores it).
* **GOAL:** human 94.55. An arm whose point is below it is reported **BELOW GOAL**, and the read prints whether its CI-hi is also below.
* ADOPT does **not** clear C1's own GUARD-FAIL (TM+C: transparent comfort, the longitudinal speed bias). Those stay open on the lineage.
* Transparent comfort is reported in both forms (the 20-pose plan, and the 8 submitted poses). It is descriptive and enters no clause.

### 5.1 Read controls (any FAIL or any missing one ⇒ INVALID, no outcome, every Δ descriptive only)
* **Carried from the parent:**
  * **CTRL-T:** the table's pick equals the harness CSV to 1e-9, on every token, for P and C.
  * **CTRL-CSV.**
  * **CTRL-SET:** 12,146 tokens over 136 logs, identical across rows, seams and dumps.
  * **CTRL-SEAM:** sha256 recorded and re-checked.
  * **CTRL-ID:** each arm read against itself gives every Δ exactly 0.0, for P and C.
  * **CTRL-DET:** **P's full seam** is re-run and must be bit-identical.
* **CTRL-ARGV, pick form.** Must hold, exactly:
  * argv(P) − argv(C) = {`--rank-w`, `--rank-heads`, `--out`};
  * C names `--rank-w 0.0` and none of `--rank-heads` / `--rank-ep-gap` / `--rank-gate-topk`;
  * P: `--rank-w` > 0, `--rank-heads` = `pick`, and neither `--rank-ep-gap` nor `--rank-gate-topk`;
  * every arm: one `--init-from`, `--steps 1000`, its own `--out`;
  * argv(Xs) − argv(X) = {`--seed 1`, `--out`}.
* **CTRL-W0**, for C and P.
* **CTRL-S0 (full form, C's side = `evidence/metrics_scorerft_c.jsonl`, sha256 `97a8631c…`), CTRL-ORDER, CTRL-CODE, CTRL-HARNESS, CTRL-CREUSE:**
  as in §3.1.
* CTRL-ROUTE is a **launch** control (§4.4). The read cannot see gradients, so it does not re-check CTRL-ROUTE. It requires the PASS token that
  carries it, through the launch gate.
* **CTRL-WRULE:** the banked weight-rule JSON reads SELECT, with heads `["pick"]`, no dead head and a real backend, and its `rule.rank_w` equals P's
  `--rank-w` **exactly**.
* **CTRL-CLIP (read form).** The banked probe JSON must satisfy all of the following:
  * its sha256 equals `CLIP_PROBE_SHA256`, and V1 to V4 read PASS;
  * its P-form argv equals P's run argv, and its C-form argv equals `argv_c.json`, each except `--out`;
  * the verdict line prints its state: `CLIP-0`, `CLIP-SAME occ=a/8`, or `CLIP-T on=… off=… occ_P=…/8 occ_C=…/8 cmin_P=… cmin_C=…`.

  A missing probe makes the read INVALID. `READ_TOOL_SPEC.md` still needs a matching `ctrl_clip` and a `--clip-probe` input (Q-P17).

### 5.4 Power (ESTIMATED)
**How it was derived.** The paired SE comes from the Stage 1 T − C CI on the **same** 12,146 tokens and 136 logs, the **same** C and the same estimator:

SE = (CI-hi − CI-lo) / (2 × 1.960)

The minimum detectable effect is MDE = (z₀.₉₇₅ + z_power) × SE, using a normal approximation to the percentile bootstrap. Source: `read_stage1.json`
fields `T_minus_C.*.tracked` and `families.TACTICAL.(b)_selection_quality.T_minus_C`.

| P − C read | Stage 1 half-width | SE | MDE at 50 % power | **MDE at 80 % power** | 90 % |
|---|---|---|---|---|---|
| **P1: tracked PDMS** | 0.307 | 0.157 | 0.31 | **0.44** | 0.51 |
| G1 NC / DAC / EP / TTC / C (size of a true adverse effect flagged) | 0.23 / 0.25 / 0.34 / 0.49 / 0.014 | 0.118 / 0.127 / 0.174 / 0.250 / 0.007 | — | 0.33 / 0.36 / 0.49 / 0.70 / 0.020 | — |
| M3 regret | 0.316 | 0.161 | 0.32 | 0.45 | 0.52 |
| zeroed-pick rate (pp) | 0.291 | 0.149 | 0.29 | 0.42 | 0.48 |
| M4 oracle-of-64 (damage) | 0.061 | 0.031 | 0.06 | 0.09 | 0.10 |

* ⇒ **P1 has ~80 % power for a true P − C of +0.44 points**, which is **6.6 % of C's regret** (6.67). Below about +0.31 it is more likely than not to
  miss.
* **Assumptions.**
  * P changes C's picks about as much as T did; T moved top-1 by +1.6 pp and zeroed picks by +1.26 pp. A P that changes fewer picks gives a narrower
    paired CI, and therefore a smaller MDE.
  * Logs only. **Training variance is unmeasured on this rig**: F was never measured, and Cs never ran. On another rig, the tiny-rig replicate read
    *separated* in 9.5 % of family cells for two runs that differed in nothing (`H-ESTIM-SEED-1`, CLAUDE.md, INHERITED, NOT imported).
* **P3's power cannot be estimated:** F is unmeasured.
* **Routing (decision 10) does NOT change these numbers.** They describe the read: the tokens, the logs, the estimator, and how many picks differ
  between the two arms. They do not describe the lever.
  * Routing masks part of the gradient, so P may move fewer picks than T did. That narrows the paired CI and **lowers** the MDE. The 0.44 figure is
    the conservative statement either way.
  * ~~Routing **does** shrink the term's gradient norm at w = 1~~. That was an ESTIMATE, and it is **contradicted** on the shared parameters.
    * The o3b implementer reports the routed gradient **larger in net** on the shared `score_head`, because the pick's downward push no longer
      cancels the better proposals' upward push (INHERITED, dev-box synthetic smoke).
    * The rule measures on parameters. So the direction of the w\* shift is unknown, and both K2 (PL1) and K2b (PL2) stay possible. The rule on the
      pod decides (Q-P2).
    * This is a weight-rule effect, not a power effect. The clip changes none of §5.4's numbers either (§3.2).

## 6 · The four metric families (MM decision 6, binding; per family, never pooled; P − C and Ps − Cs, same tokens, same paired estimator)
| family | metrics (instrument) | notes |
|---|---|---|
| **LONGITUDINAL** | speed_bias, speed_mae, speed_rmse, along_mae, along_final_bias, accel_mae, progress_ratio, progress_error, under_progress_rate, target speed within 0.5 / 1 / 2 m/s, **min_ttc_s**, and **headway / time-gap to the lead agent**, plus lead_viol, lead_at_risk, lead_progress_ratio | Computed on the tracked-executed frame by the chain `p5c_families.py` → `families_run.py --per-window` → `fam_ci.py` (paired). Lead rows are computed where a lead exists, with their n, and refused with the reason where none does. The S1Z failure mode (slower picks) is read here. Polarity follows the ONE shared rule, `field_read.entry_adverse_pol`: min TTC, headway and time-gap are higher-is-better. |
| **LATERAL** | heading error, **curvature error, yaw-rate error**, cross_mae, cross_bias, cross_final_mae, last-pose heading error | as in Stage 1 |
| **TACTICAL** | (a) manoeuvre decision: turn-active command adherence + κ (`p5c_tactical.py`, relabelled C1→P, C0→C), and the `fam_ci` lateral / longitudinal / 5-way decision accuracy + κ. (b) **selection quality from each arm's 64-proposal table**: regret and its gate / EP / TTC / C / other split, top-1, zeroed-pick rate (all, by DAC, by NC), TTC-0 pick with a TTC-1 alternative, within-token AUC NC / DAC / TTC, EP Spearman over safe proposals, the confidently-wrong DAC / NC shares. (c) **tactical goal-setting** | (c) is **NOT COMPUTABLE**: REFe has no tactical goal head (n = 12,146, reason stated) |
| **STRATEGIC** | **In full** (PI unpaused 2026-11-16), from `strategic_family.py read --arm P=<P seam> --base C=<C seam>`. S_DEC: all-token accuracy and κ, turn-active accuracy and κ, per class LEFT / STRAIGHT / RIGHT. S_GOAL per command: endpoint error, endpoint heading error, route-progress ratio, lateral offset, \|lateral offset\|. Its controls: logged future as an arm, mirror P / C, self vs self, G_COUNTS. | No bar. ⛔ The read never prints PAUSED, and it refuses a strategic JSON that says PAUSED |

* A separated adverse cell in any family is **stated in the verdict line as a qualifier**. Only G1 blocks ADOPT; that was decided here, before data.
* ⛔ **Every separated-adverse row is listed** in the verdict note with its CI and n (the PI's every-row rule). A count is not enough; Stage 1 listed all 25.

## 7 · Mechanism reads (MM decision 5; they decide only WHICH next arm, never ADOPT; P − C, the same 10,000 shared draws, seed 20260927)
For a decision-level term, the primary mechanism read is the decision's own loss: the regret.
* **M3, regret** (oracle-of-64 − pick, from each arm's 64-proposal table). **Moved** iff CI-hi < 0.
  * Reported beside it, with CIs: the components gate / EP / TTC / C / other (`regret_decomp.py` split, re-stated per token in the read) and the top-1
    rate.
* **M2, zeroed-pick rate** (pp), plus zeroed-by-DAC, zeroed-by-NC and the confidently-wrong DAC / NC shares.
  * **Z↑** := zeroed-pick rate P − C has CI-lo > 0, meaning P zeroes more picks. Z↑ selects OP6's branch (§9) and nothing else.
  * Under routing, the term cannot lower a safe pick's gate output. A Z↑ under P therefore points at the residual path of §1.1: shared weights that
    carry a live gate gradient from one proposal to the outputs of others.
* **M4, generation guard** (oracle-of-64). **Damaged** iff CI-hi < 0.
* **M1, EP Spearman** over safe proposals: **reported, NOT decisive.** Its "moved" flag is printed and enters no outcome.
* **M5, train-side monitor** (descriptive, zero GPU): P's logged navtrain `pick_v1 / best_v1 / skill_v1` from its own `metrics.jsonl`, as a trend from
  step 0 to step 999. It distinguishes "the term optimised the training-side decision but it did not transfer" from "it did not optimise". C logs no
  `*_v1` keys (its rank term is `None`), so the comparison is against P's own step 0. It enters no outcome.
* **mechanism moved := M3 moved.**
* ⚠️ **Identity, stated before data.** Per token, Δregret = Δoracle − Δpick. So, when M4 is not damaged, M3 can move without P1 only if the oracle
  drifts down without separating, or if the two CIs straddle their bars. OP5 is the name for exactly that case.

## 8 · Seed replicate plan (`H-ESTIM-SEED-1`)
* **ADOPT requires the seed-1 pair (P2) and the floor clause (P3).** Ps and Cs differ from P and C only by `--seed 1` (data order and augmentation
  draw; same init).
* **Cs is the parent card's never-run Cs.** Its F is also the first measured control floor of the SCORER-FT rig.
* **Inference seed:** none. The argmax is deterministic, and CTRL-DET re-checks it on P.
* A Stage-1 FAIL (OP3 to OP7) is accepted **on one seed**, decided before data, so that the replicate is spent only where an effect could be real. The
  verdict states that training variance was not measured for that arm.
* Where more evidence is needed (OP2, OP5), the cheapest next experiment is **seeds, not windows** (CLAUDE.md).

## 9 · Outcomes (committed now; exhaustive and exclusive; evaluated IN THIS ORDER) and the NEXT arm for each (Rule Zero)
**Pre-launch (the weight rule on P's argv):**

| # | condition | verdict | next arm, committed now (cost ESTIMATED) |
|---|---|---|---|
| **PL1** | **K2:** r(4) < 1, or `pick` in `dead_heads` | **NOT LAUNCHED: the pick pull is too weak in aggregate space** | **P-log.** Same pick, same pairs, same weights, with z = log `navsim_v1`(sigmoid(s)) = logsigmoid(NC) + logsigmoid(DAC) + log(5σEP + 5σTTC + 2σC) − log 12, with §1.1's routing applied inside. The margin then lives in log space, as the parent's `gate` z does. Own card and own weight rule; C reused; ~14.2 L40-h. PI go |
| **PL2** | **K2b:** ρ(w*) > 1 | **NOT LAUNCHED: the term would out-pull l_traj on the trunk** | **P-detach.** `pick` on a second scorer pass over a **detached** `visual_ctx`, new flag `--rank-detach-ctx` (the parent's O6 arm). ρ ≡ 0 by construction. Own card; C reused; ~14.2 L40-h. PI go |
| **PL3** | **CTRL-CLIP INVALID** (§3.2): any of V1–V4 FAILS, or the probe did not run | **NOT LAUNCHED until the probe is re-run and valid** | Re-run the probe (~0.4 L40-h). **A V2 or V3 failure means C is not reusable ⇒ C′** (§3.1's committed response, ~14.2 L40-h, PI go) |
| **PL4** | **CLIP-T** (§3.2): n_on + n_off > 0 | **LAUNCHED, tagged CLIP-T: P − C is read as the effect of the `pick` term at w\*_P under the recipe's global clip at 1.0** | P runs as planned. The tag's consequences are fixed now: (a) every verdict line carries the CLIP-T string of §5.1; (b) no verdict attributes P − C to the term's gradient *as the weight rule sized it*; (c) the OP4–OP7 verdict notes name the clip as an unexcluded co-mediator; (d) OP1–OP7 keep their conditions and their next arms; (e) the register row records CLIP-T |

CLIP-SAME or CLIP-0 ⇒ launched, with no tag beyond the CLIP state the verdict line always prints.

**Read states (no outcome):**

| state | condition |
|---|---|
| **INVALID** | any read control FAILS or is missing. If the failure is in CTRL-S0, CTRL-ORDER or CTRL-CODE, the next step is C′ (§3.1). |
| **INCOMPLETE** | ¬P1 and no 64-proposal tables: OP4 to OP7 cannot be told apart. |

**Outcomes, in order:**

| # | condition | verdict | next arm, committed now (cost ESTIMATED) |
|---|---|---|---|
| **OP1** | P1 ∧ G1 ∧ P2 ∧ P3 | **ADOPT** (BELOW GOAL unless ≥ 94.55) | `pick` becomes the scorer term of the C1 recipe and carries into the next generator arm (TEACHER-N7 / comfort). **BELOW GOAL ⇒ dose:** P₂ₖ vs C₂ₖ, 2,000 steps from C1 (C1 parity, the parent's named alternative), own card, ~57 L40-h, PI go. **AT GOAL ⇒** C1's GUARD-FAIL (TM+C) becomes the lineage's top item |
| **OP2** | P1 ∧ G1 ∧ ¬(P2 ∧ P3) | **SWAMPED, NOT PROVEN** | One more seed pair: P s2 and C s2, ~28.4 L40-h. Seeds, not windows. Still swamped ⇒ OP7's next. PI go |
| **OP3** | P1 ∧ ¬G1 | **ADVERSE, NOT ADOPTED: the pick bought PDMS with a guarded sub-score** | **P-guard:** pick pairs admitted only where y_j^h ≥ y_k^h for **each adverse sub-score h** in the G1 list (NC, DAC, EP, TTC or C), new flag `--rank-pick-guard <h,…>`, so that the term can no longer pay with h. Own card; C reused; ~14.2 L40-h + weight rule. PI go |
| **OP4** | ¬P1 ∧ M4 damaged | **FAIL: the trunk path hurt generation** | **P-detach** (as PL2): rank stops at the scorer decoder, and the BCE path is unchanged. C reused; ~14.2 L40-h. PI go |
| **OP5** | ¬P1 ∧ ¬M4 damaged ∧ M3 moved | **FAIL: the decision moved toward the oracle, PDMS did not clear its bar** | **Ps + Cs at seed 1** as a measurement, ~28.4 L40-h. Their read is governed by a **successor card frozen before Ps / Cs are read**, with a pooled two-seed bar and F; **this card cannot adopt on OP5**. Cs is the rig's first floor in any case. PI go |
| **OP6** | ¬P1 ∧ ¬M4 damaged ∧ ¬M3 moved ∧ Δ₀ CI-hi < 0 | **FAIL: the decision term made the pick WORSE** | **Z↑ ⇒ P-top8:** the same routed, gap-weighted term, anchored on **every proposal in the scorer's own top-8** by `navsim_v1` (the registered `SCORERFT_RANK_GATE_TOPK` = 8 set), not on the argmax alone. Pairs (j, i): i in the top-8, PDMS_j > PDMS_i. Routing per pair: the anchor's factors are live only where y < p, the better proposal's only where y > p. Same normalisation. New head key `pick8`; own card; C reused; ~14.2 L40-h + weight rule; PI go. Rationale and rejected alternatives: `CARD_QUESTIONS.md` Q-P14. The earlier gate-detached arm is **subsumed** by decision 10. **¬Z↑ ⇒** OP7's next. M5 is reported beside the verdict |
| **OP7** | otherwise (¬P1, nothing moved, not worse) | **FAIL: scorer training does not carry the selection margin, at head level (T) or decision level (P)** | **The selection line closes** (the parent's O4 next): generation **TEACHER-N7** (`PREREG_P5C_NEXT` §5 rank 3), **or** explicit proposal-geometry inputs to the EP head (architecture: PI). **Both are named blockers on the PI and the L40** |

* **G1 with ¬P1.** An outcome in OP4 to OP7 that also fails G1 says so in the verdict line (`G1=FAIL adverse=…`), as Stage 1 did. It does not change
  the outcome.
* **Rule Zero.** Every row ends in a run or in a named blocker. OP7's blockers are the PI (architecture or generation go) and the L40 (queue).
* **Routing carries forward.** Every next arm built on `pick` inherits decision 10's routing and CTRL-ROUTE: P-log, P-detach, P-guard, P-top8, the
  seed pairs and the dose pair.
* **CTRL-CLIP carries forward too.** Each of those arms runs its own probe against its own control before launch.
  * In Stage 2 the forms are argv(Cs) and argv(Ps), and V2 and V3 compare the Ps-form's rows with the Cs-form's.
  * The verdict line of Ps − Cs prints that probe's CLIP state.

## 10 · Multiplicity, snooping, ledger
* **Tests.** One primary test (P1) plus its replicate (P2), and five guards (G1). The families and M1 to M5 are descriptive, carry CIs, and adopt nothing.
* **Ledger.**
  * P's Stage-1 read (P vs C) is **look 1 of `H-SCORERFT-PICK-1`**. Ps vs Cs, if run, is **look 2**.
  * **C's navtest rows were already read** at look 1 of `H-SCORERFT-1` (C = 92.67). The two primary tests T − C and P − C **share their control**, so
    they are correlated.
  * Multiplicity across cards is not corrected, and the verdict says so.
* **Snooping disclosure.**
  * The `pick` lever was committed in the parent card **before any SCORER-FT data**. The parent's heads and sizes were chosen from C1's SNOOPED navtest
    diagnostics.
  * The implementation details were fixed **after** the Stage 1 numbers were seen. None of them was fitted to navtest:
    * strict inequality;
    * float32 pick;
    * the constant-batch normalisation (the programme's existing convention, MM decision 1);
    * no temperature (MM decision 2);
    * **label-consistent routing (MM decision 10)**. It was motivated by Stage 1's navtest numbers: T's regret_gate +1.27 and its zeroed-pick rise. It
      was decided before any P data, and it contains no number chosen on navtest.
    * **CTRL-CLIP (§3.2)**, prompted by o3b's dev-box smoke, decided before any P data. Its numbers all come from elsewhere:
      * K = 8 is the weight rule's `N_MICRO`;
      * the threshold is the trainer's own 1.0 and torch's own coefficient;
      * the comparison is exact, per step.

      It decides no adoption.
  * w*_P comes from a navtrain-only rule. P trains on navtrain only, so the navtest read tests a trained model, not a rule fitted on navtest.

## 11 · Cost (ESTIMATED unless marked) and the PI decisions requested
* **Stage 1.** P = 1,000 × 51.2 s = **14.2 L40-h**. The step cost is MEASURED on C; the pick term adds a [B, 64, 64] mask and an aggregate per set,
  ESTIMATED negligible. Add the weight rule (~0.5 L40-h), the gate smoke (~0.5 L40-h) and CTRL-CLIP: ~0.4 L40-h, from 2 × 417.5 s MEASURED on C's
  steps 0–7 plus two builds (ESTIMATED). Total **~15.6 L40-h**.
* **Stage 2, conditional on P1 ∧ G1:** Cs + Ps = **28.4 L40-h**, plus ~0.4 for its CTRL-CLIP.
* **Dev box (CPU, plus the dev-box GPU lock for seams):**
  * P's seam: C's seam report records 17,677 s; the runbook quotes 21,175 s for C's `s2` step.
  * P's full re-run seam for CTRL-DET: ~5 h.
  * The 64-proposal table, one transp_score pass (P, C, base), families, tactical, strategic and the read.
  * ~14 h of wall-clock in total, UNVERIFIED.

**PI decisions REQUESTED (none given yet):**
1. Stage 1: P (~15.6 L40-h, CTRL-CLIP included), queued on the L40 at the MM's position.
2. Stage 2: Cs + Ps (+28.8 L40-h, CTRL-CLIP included), **conditional** on P1 ∧ G1.
3. Anything else (an extra seed pair, any next arm in §9) needs a new PI go.

## 12 · Proposed register row (the MM writes it from the tip, in the same turn it freezes this card)
`H-SCORERFT-PICK-1, pre-registered`. Fields:
* **Provenance:** <date>, MM; `PK/raw/2026-11-24-scorerft-pick/PREREG_SCORERFT_PICK.md`, sha256 in LOOP_STATE.
* **Question:** does the decision-level `pick` term (every proposal the label PDMS ranks above the scorer's own `navsim_v1` pick, paired against that
  pick, weight = the PDMS gap, through the trunk, with label-consistent gradient routing, MM decision 10) raise tracked PDMS over the reused
  SCORER-FT control C?
* **Status:** OPEN, frozen before the weight rule or any P step.
* **Arms:** P (seed 0) vs the reused C, with conditional Ps / Cs (seed 1), 1,000 steps from C1.
* **Bar:** P1 CI-lo > 0, G1 no adverse sub-score, P2 replicate, P3 d > 2F.
* **Controls:** the parent's, plus CTRL-S0 (full form; PASS on T vs C at Stage 1, 19 + 10 keys), ORDER, CODE, HARNESS, CREUSE, WRULE, the launch
  control CTRL-ROUTE, and the pre-launch probe CTRL-CLIP (8 real steps, C-form vs P-form, on the trainer's global clip at 1.0; its CLIP state is
  recorded in the row).
* **Outcomes:** OP1 to OP7 (plus PL1 to PL4), each with its next arm.
* **Origin:** the parent's O3 (`H-SCORERFT-1`, Stage 1: heads rank, the pick does not).

## 13 · Known residue (MM decision 9) and the retired arms
* **T / Ts are RETIRED.** The Stage 1 read is O3, so there is no Stage 2. The T model stays banked (HF `scorerft/t`) as the record of `H-SCORERFT-1`
  look 1, and Ts is never trained.
* **The census pin of T / Ts is stale and is NOT re-pinned.**
  * `SCORERFT_RANK_CENSUS["rank_loss_sha256"]` = `c0e027d2…` is the sha256 of `rank_loss.py` at `83411c28` = `9d11eb0f` = `5eb74f30` (MEASURED).
  * After the P land, `rank_loss.py` is `3f782697…`.
  * A relaunch of T / Ts would fail G-LIVE L2 whenever its smoke sees zero `gate` pairs (INHERITED, o3 README Q3). They are retired, so no relaunch is
    planned.
* **The pin does not touch this card's arms.** P carries no census pin (the README says `pick` forms pairs on almost every set, so L2 is met by the
  smoke), and C / Cs carry none (control arms).
