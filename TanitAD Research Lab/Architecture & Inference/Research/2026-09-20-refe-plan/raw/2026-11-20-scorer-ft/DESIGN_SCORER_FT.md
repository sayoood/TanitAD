# DESIGN · SCORER-FT: a within-token ranking term for the REFe scorer, trained through the representation

Written 2026-11-20 by the SCORER-FT design agent for the Master Mind. **DRAFT ONLY.** Nothing is built, launched, staged,
committed or pushed. Every line number refers to commit `e17d040e`, read with
`git -C C:/Users/Admin/tanitad-push show e17d040e:<path>`. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
The card that goes with this design is `PREREG_SCORER_FT_DRAFT.md` (same folder).

---

## 0 · Headline (read this first)

1. **No within-token ranking term exists in the trainer C1 was trained with today.** `PK/refe/train.py` has one scorer
   term: a pointwise BCE on the 6 heads, weighted equally (§1.3). Ranking terms do exist, but **only in the heads-only trainer**
   `PK/refe/scorer_finetune.py` (ListNet / expected reward, SCORER1 `PairLoss`, S1Z twin, EPO). That trainer freezes the
   representation (`scorer_finetune.py:3-6`) and never ran on C1. On P6 every arm in that family was **swamped by training noise**
   (§1.6). ⇒ The design is **not moot**, but what is genuinely new has to be the **path the term takes (through the trunk)**, not
   the loss family alone. (The frozen set is `scorer_finetune.py:92` `SCORER = ("score_q_mlp", "score_dec", "score_head")`.)
2. **The EP head regresses an unconditional, saturating soft label.** It is NAVSIM v1.1's pairwise EP against PDM-Closed on the
   LQR-tracked execution. It is **exactly 0 on every proposal with NC·DAC = 0**, and **1 for every proposal that reaches PDM-Closed's
   progress** (§1.4). The label equals the harness sub-score. But the planner's `navsim_v1` aggregate multiplies it by pNC·pDAC
   again, so the safety factor is charged twice.
3. ⚠️ **The brief names the wrong labeller.** `onpolicy_label_v4.py` is the old OnPolicyBank route, with EP measured "relative to the
   same teacher advance" (`onpolicy_label_v4.py:11-13`). **P5-C does not call it.** P5-C runs `--labeller online`, which goes through
   `online_labeller.py` → `navsim_pdm_targets.score_plans` (§1.2). The labels SCORER-FT uses are those online labels, and **they
   agree with the harness** (§1.5). ⛔ Do not build SCORER-FT labels with `onpolicy_label_v4.py`.
4. **Design:** one new term `L_rank` inside `train.py`'s online deferred backward. It uses **pairwise logistic losses within each
   64-proposal set**, weighted by the harness PDMS gap:
   * **EP pairs** are formed among SAFE proposals only (attacking the EP regret, 2.63);
   * **gate pairs** put a safe proposal against an unsafe one that sits in the scorer's own top-k (attacking the boundary DAC/NC
     zeroed picks, 2.72).

   It fine-tunes **from C1, heads plus representation** (`--scorer-sees-trunk`, the C1 recipe unchanged otherwise). It needs no new
   labels, no model change and no labeller change.
5. **Cost (ESTIMATED):** 1,000 steps × ~51 s/step = **14.2 L40-h per arm**. Stage 1 (treatment + control) is **28.3 L40-h**. Stage 2
   (the seed-1 replicates, conditional on Stage 1) is another **28.3 L40-h**.

---

## 1 · Step 1: how the scorer is trained today (source trace, file:line at each hop)

### 1.1 Launch → trainer
| hop | file:line | what it says |
|---|---|---|
| profile | `PK/refe/launch_gate.py:246` | `_BASE = dict(trainer="refe/train.py", ...)`. The P5-C profiles are cloned from `refe_p6` (`:475-505`), which is `{**_BASE, ...}` (`:270`) |
| argv | `PK/refe/launch_gate.py:457-472` `p5c_argv` | `--scorer-sees-trunk --yaw-wrap off --nav-mode command --labeller online --label-workers 13 --label-chunks 2 --init-from <P7 base> --epochs 0 --steps 2000 --batch 4 --accum 64 --lr 5e-5 --weight-decay 0.01 --amp bf16 --score-w 0.1 --w-field 0.0 --ego-yaw-table …`, plus (C1 only) `--traj-label-table`. There is **no scorer-ranking flag** |
| live terms | `PK/refe/launch_gate.py:60` | `P6_TERMS = ("traj_L1","score","pos_L1","yaw_L1","bce_nc","bce_dac","bce_ep","bce_ttc","bce_c","bce_ddc")`. G-LIVE knows of no ranking term |

### 1.2 The labeller the P5-C trainer actually calls
| hop | file:line | what it says |
|---|---|---|
| flag | `PK/refe/train.py:2245-2250` | `--labeller {bank,online}`; online = "NAVSIM PDM targets on each sample's OWN current proposals … lag 0" |
| construction | `PK/refe/train.py:1535-1542` | `import online_labeller as OL`; `OL.OnlineLabeller(backend=a.label_backend, …, n_prop=cfg.n_proposals, horizon=cfg.horizon_steps, chunks=a.label_chunks)` |
| submit | `PK/refe/train.py:1810-1813` | the micro-batch's own detached proposals `traj.detach()` → `lab.submit(micro, step, metas, props_np)` |
| worker | `PK/refe/online_labeller.py:112-135` `_job` | `labels_from_targets(PT.score_plans(parts, PT.executed_plans8(props, repair=True)))` on the D9 metric-cache parts |
| transform | `PK/refe/navsim_pdm_targets.py:132-139` | `executed_plans8`: the A7 last-heading repair, then `navsim_dac.to_navsim`. This is "the SAME transform the planner uses" (`online_labeller.py:11`) |
| scoring | `PK/refe/navsim_pdm_targets.py:99-129` `score_plans` | `[PDM-Closed, *proposals]` → `simulate_proposals` (NAVSIM's LQR + bicycle) → `PDMScorer.score_proposals`; per-proposal NC, DAC, EP, TTC, C, DDC; PDM-Closed (index 0) dropped |
| order + clip | `PK/refe/online_labeller.py:39, 73-75` | `COMPONENTS = ("nc","dac","ep","ttc","c","ddc")`, clipped to [0, 1] |
| identity guards | `PK/refe/train.py:1911-1916` | sha256 of the scored proposals == the trainer's own copy; label stamp == micro-batch, lag == 0 (else raise) |

`onpolicy_label_v4.py` is reached only through `OnPolicyBank` / `--scorer-mode onpolicy` (bank mode). `train.py:1252` refuses that
mode together with `--labeller online`.

### 1.3 The loss on the 6 heads (the only scorer term)
| hop | file:line | what it says |
|---|---|---|
| model | `PK/refe/model.py:563` | `score_head = nn.Linear(dec_width, 6)`; the six PDM components |
| scorer input | `PK/refe/model.py:779` | `s = self.score_q_mlp((traj.detach() if detach_scorer else traj).flatten(2))`. The trajectory is detached in the default forward |
| context | `PK/refe/model.py:784`; flag `PK/refe/train.py:2173` | `sctx = visual_ctx.detach() if cfg.detach_scorer_context else visual_ctx`. **`--scorer-sees-trunk` sets it to False, so the score loss reaches `scene_proj` and the backbone LoRA** |
| set context | `PK/refe/model.py:703-704` | the scoring decoder's self-attention spans the 64-proposal set. This is an architectural set context, **not a ranking objective** |
| forward | `PK/refe/train.py:1785` | `traj, score = model(img, ego, goal, calib=…)` (online path) |
| **loss** | `PK/refe/train.py:1919-1922` | `bce = F.binary_cross_entropy_with_logits(P["score"], lab_t, reduction="none")  # [B, M, 6]`; `per = bce.mean(-1)`; `l_score = (per * msk[:, None]).sum() / (cov_norm·B) * score_w` |
| weights | `PK/refe/train.py:2193` | `--score-w` default 0.1 (P5-C names 0.1); the 6 heads are weighted **equally** (`mean(-1)`), DDC included although `navsim_v1` ignores it |
| backward | `PK/refe/train.py:1938-1943` | `tot_ = l_traj + l_score (+ w_field·L_field + aux)`; `(tot_ / accum).backward()` |
| monitor | `PK/refe/train.py:1923-1935, 2011-2019` | `torch.no_grad()`: pick / random / best / skill from `_pdm_aggregate`. No gradient |

**Answer to Q1.** The loss is **pointwise only**: soft-target BCE per proposal per head, heads equal-weighted. There is **no
pairwise or listwise term** and **no within-token normalisation**. The ranking the planner needs is learned only as a side effect of
calibrating each proposal on its own.

### 1.4 What the EP head regresses (`navsim_pdm_targets.py:112-122`)
`raw = progress_raw × mult` with `mult = NC·DAC` (the v1.1 multiplicative metrics). The label is
`ep_j = raw_j / max(raw_PDMClosed, raw_j)` if that max exceeds 5 m. Otherwise it is 1 when `mult_j ≠ 0`, and 0 when not. ⇒ it is
pairwise against PDM-Closed (not the 64-set max), **0 whenever NC·DAC = 0**, **saturated at 1** for every proposal that reaches
PDM-Closed's progress, and halved under NC = 0.5. The head is trained toward this target by BCE (§1.3).

### 1.5 Do the labels agree with the harness? Yes, with two caveats
* **Agreement (by formula, plus a sibling measurement).** The labeller uses NAVSIM's own simulator and scorer, the planner's own pose
  transform, and the same pairwise EP against PDM-Closed. The navtest per-proposal instrument `p5c_prop_score.py` (via
  `transp_score.score_arms`, a sibling implementation with the same pairwise EP; header lines 4-5) reproduced the harness CSV
  for the pick on every token (CTRL-T, D-REFE-SRULE-1). ⚠️ It is a sibling, not the same file. A labeller-vs-harness identity on
  NAVTRAIN tokens is **UNVERIFIED** here (it is a G-LIVE row below).
* **Caveat 1, use not value: the double charge.** `planner.py:558-559` (`DEFAULT_RULE = "navsim_v1"`, `:525`) ranks by
  pNC·pDAC·(5pEP+5pTTC+2pC)/12. A BCE-trained pEP estimates E[EP] = P(safe)·E[EP|safe], so P(safe) enters the EP term twice. This
  was diagnosed on P2 (`D-REFE-EP-2`; pre-registered fix `H-EPR1`, **never run**). The design forms EP pairs among safe proposals
  only, so `L_rank` trains the conditional ordering. The BCE label is left unchanged (one lever per arm).
* **Caveat 2, the monitor's aggregate.** `train.py:616-619` `_pdm_aggregate` is the **v2_shape** rule,
  NC·DAC·DDC·(5EP+5TTC+4C)/14. It is not the `navsim_v1` the planner deploys and the harness scores. So the log line
  `on-policy: sets 256 pick 0.928 random 0.688 best 0.973 skill 0.842` (`train.py:2018`; skill = (pick−random)/(best−random)) is
  **not in the harness's aggregate**. `L_rank` uses `navsim_v1`, and the build adds `*_v1` monitor fields (no gradient).

### 1.6 What the heads-only ranking family already measured (why the representation must move)
All **MEASURED, T1-class, full navtest 12,146 tokens, paired log-cluster bootstrap 10,000 draws** (GAC rows quoted from `e17d040e`):
* `H-SCORER1P6-1` **O5 SWAMPED**: S1E (pair loss on NC/DAC/TTC + EP) +0.074 [−0.127, +0.272] vs P6. C0r − C0 floor F = 0.380.
* `H-S1Z-P6-1` **Z5 SWAMPED**: twin loss +0.125 vs C0, below 2F. EP **−0.449 [−0.642, −0.249]** adverse ("pays … by picking slower plans").
* `H-S1ZSM-P6-1` **SM5**: EP not recovered. `H-REP-P6-1` **REP-B NOISE**: every 400-update fine-tune lifts P6 by +0.2 to +0.3, and no
  auxiliary loss adds anything over plain fine-tuning (s_p 0.179).
* `H-DOSE-P6-1` **DOSE-C SATURATED** at 1,600 updates. The card named the next step: "leave the scorer fine-tune line".
* `D-REFE-SRERANK-1` (C1): frozen outputs plus proposal geometry carry no margin (re-ranker B switches 14/6,251 tokens, Δ +0.015
  [−0.007, +0.050]). Its committed next step: **"SCORER-FT must reach the representation (not heads-only)"**.

⚠️ Two differences from that family, and both are needed for this to be a new experiment rather than an eighth repeat:
(i) the gradient reaches the **trunk**; (ii) the labels are **lag-0 online labels on the model's own current proposals**, not a
frozen bank. And the EP pairs are **safe-only**, which S1E's `PairLoss` did not enforce: it has no safety condition
(`scorer_finetune.py:176-270`, `PairLoss`), so its EP pairs also taught safety.

---

## 2 · The change: `L_rank` (the smallest change that attacks the measured terms in size order)

### 2.1 Measured targets (MEASURED, `PK/raw/2026-11-20-p5c-next/propscore/`, T1 LQR-tracked, all 12,146 navtest tokens, SNOOPED population)
| term | size (PDMS ×100) | head diagnosis | `L_rank` component |
|---|---|---|---|
| zeroed-pick gate | **2.72** (357 tokens) | DAC/NC AUC 0.96 / 0.93 on average, but on rescuable zeroed picks the pick's DAC output is above the passing proposals' median on **70.2 %** of 198 (NC 58.7 % of 138) (`HEAD_AUC.json`) | **gate** (boundary, top-k) |
| EP | **2.63** (11,780 linear tokens) | EP within-token Spearman over safe proposals **0.413 [0.389, 0.434]** vs plain path length **0.828** (`EP_GEOM_CHECK.json`) | **ep** (safe-only) |
| TTC | 1.38 (414 tokens: TTC-0 pick with a TTC-1 alternative) | AUC 0.900 | `ttc` built, **not in the primary arm** (§2.6) |
| C / other | 0.00 / 0.06 | — | none |

`ep`+`gate` cover 5.35 of the 6.80 regret (79 %). The EP head has the largest learnable gap: the information is in the proposal's
own geometry, which the scorer already reads (`model.py:779`).

### 2.2 Definition (per labelled set b; logits `s ∈ R^{64×6}` in order NC, DAC, EP, TTC, C, DDC; labels `y` from §1.2)
```
safe_m   = (y_m,NC > 0.5) & (y_m,DAC > 0.5)                      # = NC·DAC = 1; same as head_auc.py:54
pdms_m   = y_NC·y_DAC·(5·y_EP + 5·y_TTC + 2·y_C)/12              # navsim_v1 (planner.py:558-559), the harness aggregate
agg_m    = navsim_v1(sigmoid(s_m))  [no_grad]                    # the planner's own pick score
topk     = indices of the k largest agg (torch.topk, k = --rank-gate-topk)

EP   pairs: {(i,j): safe_i & safe_j & y_i,EP − y_j,EP ≥ g}       weight a_ij = (5/12)·(y_i,EP − y_j,EP)   z = s_EP
gate pairs: {(i,j): safe_i & !safe_j & j ∈ topk & pdms_i > pdms_j} weight a_ij = pdms_i − pdms_j          z = logsigmoid(s_NC) + logsigmoid(s_DAC)
TTC  pairs: {(i,j): safe_i & safe_j & y_i,TTC = 1 & y_j,TTC = 0}  weight a_ij = 5/12                       z = s_TTC      (only if named)

L_h(b)  = Σ a_ij·softplus(−(z_i − z_j)) / Σ a_ij      (0 if the set has no pair of head h; counted)
L_rank  = (1 / --batch) · Σ_b msk_b · Σ_{h ∈ heads} L_h(b)
tot_    = l_traj + l_score + rank_w · L_rank (+ field / aux as today)        # inside the deferred backward, train.py:1938
```
* **Weights = harness PDMS gaps.** The term spends its gradient where the harness loses points, so it attacks the terms in
  proportion to their measured size without a hand-set per-head weight.
* **Per-set weighted mean, constant denominator `--batch`.** This is the accumulation lesson of `train.py:1858-1871`: a per-micro-batch
  own-count denominator gives `mean_i(S_i/n_i)` and broke `--accum` before. With a constant B, the accumulated gradient equals the
  full-batch gradient exactly (a G-SUITE test).
* **Gate restricted to the scorer's own top-k.** An unsafe proposal ranked low by the aggregate costs nothing. The boundary cases are
  the unsafe proposals the pick could become.
* `z` for the gate is the log of the aggregate's own gate factor, so the gradient reaches the NC and DAC logits and nothing else.
* Every component trains only rows {NC, DAC, EP} (+TTC if named) of `score_head`; rows C and DDC get **exactly 0.0**. The
  trajectory decoder gets **exactly 0.0** (the detach at `model.py:779`). Both are analytic zero controls for G-LIVE.

### 2.3 Fine-tune from C1, heads plus representation
* **Init:** C1's final `model_final.pt` (`/workspace/data/refe_runs/p5c_c1/` per `launch_gate.py:471`). The optimiser is **fresh** AdamW,
  and the step-0 digest equals the file, as P5-C's card 10.1. ⚠️ **New pin `SCORERFT_BASE_SHA256` must be MEASURED by the MM**, and
  the file banked (HF + sha256) before launch: it must not be the only copy.
* **Recipe = C1's argv** (`p5c_argv("C1")`) with three edits: `--init-from <C1 final>`, `--steps N` (N = 1,000 recommended, §5), and
  its own `--out`. `--scorer-sees-trunk` stays, so `l_score` and `L_rank` both reach `visual_ctx` → `scene_proj` → LoRA. `l_traj`
  on C1's label table (`3b198314…`, rank policy `rank0`) stays on, so the generator stays anchored by its own loss.
* **Why not heads-only:** §1.6. Seven heads-only arms on P6 sat inside the seed floor, DOSE saturated, and SRERANK found no margin in
  frozen outputs. A heads-only `L_rank` is a cheap optional **attribution** arm (`scorer_finetune.py`, about 400 updates), **not
  recommended**: the family is measured.
* **The cost of moving the trunk:** `L_rank` can perturb the proposals (the generator shares `visual_ctx`). So the card reports the
  **oracle-of-64** and the full families. Outcome O6 has its own next arm: the rank gradient stopped at the scorer decoder.

### 2.4 Exact new flags (`train.py`, all default OFF; OFF path = today's trainer bit for bit)
| flag | type, default | refusal (`rank_refusals(a)`, exit 4 before any build) |
|---|---|---|
| `--rank-w` | float, **0.0** | must be finite and ≥ 0. With `> 0`: needs `--labeller online` (the only path built), and `--rank-heads` non-empty. **0.0 ⇒ the term is not computed** (no extra op, the `w_field` pattern) |
| `--rank-heads` | str, **""** | a comma list ⊂ {`ep`,`gate`,`ttc`}, no duplicates; named with `--rank-w 0` ⇒ refused ("configured and never trained") |
| `--rank-ep-gap` | float, **0.02** | must be in (0, 1]; named without `ep` in heads ⇒ refused |
| `--rank-gate-topk` | int, **8** | must be in [1, n_proposals]; named without `gate` ⇒ refused |

Constants are fixed in advance and never tuned on navtest. `g = 0.02` EP is 0.83 PDMS points (5/12·0.02·100), so near-ties the
tracker could flip are excluded. `k = 8` is 1/8 of the set. Both are ESTIMATED design choices, frozen by the card. The **weight
`rank_w` is set by a committed rule** (§2.5), and the gate may only veto it.

Also in the build, no lever: `IDENTITY_ARGS` + `IDENTITY_DEFAULTS` (`train.py:1093-1104`) gain `rank_w 0.0`, `rank_heads ""`,
`rank_ep_gap 0.02`, `rank_gate_topk 8`, so a resume with another rank configuration is refused and old checkpoints read as rank-off.
There is a `rank` start event (the built rule: heads, gap, top-k, aggregate `navsim_v1`, reduce `set-wmean/const-B`). Per-log-row
fields: `rank_L`, `rank_w`, `rank_pairs_{ep,gate,ttc}`, `rank_sets_{…}`, and the no-gradient `pick_v1 / best_v1 / skill_v1`.
Deliberate-regression arms are environment mutations (`REFE_RANK_MUTATE=…`, the `REFE_LAB_MUTATE` / `REFE_PAIR_MUTATE_*`
convention), never flags. The launch profile forbids the variable.

### 2.5 The weight rule (committed now; mirrors `PK/refe/weight_rule.py` for L_field)
At C1's init, on 8 real micro-batches of the arm's bank (seeded draw, seed 20261012), with **real online labels** (pdm backend,
pod CPU), measure:
`r(w) = ‖∇_S (w·L_rank)‖ / ‖∇_S l_score‖`, with S = {`score_q_mlp`, `score_dec`, `score_head`}, gradients pooled over the 8, and
`ρ(w) = ‖∇_T (w·L_rank)‖ / ‖∇_T l_traj‖`, with T = the trunk group (LoRA + `scene_proj`).
Candidates w ∈ {1/64, 1/32, …, 4}. **Take the smallest w with r(w) ≥ 1.0.** The new term must be at least as strong on the scorer
as the BCE that already reached AUC 0.93-0.96 without ranking.
* **K2:** r(4) < 1 ⇒ not launched (the pairs are too few or too easy; report the pair counts).
* **K2b:** ρ(w*) > 1.0 ⇒ not launched (the ranking would out-pull the trajectory loss on the shared trunk). This goes back to the MM.

The tool's JSON also reports the per-head pair counts. G-LIVE consumes it (L2 below). Exit codes and `NOT A DECISION` stamping
follow `weight_rule.py`. It is a pod-run instrument (it needs the D9 metric cache and the labeller venv).

### 2.6 Why these heads in the primary arm, and not TTC
`ep` + `gate` are the two largest terms (2.63, 2.72) and the brief's order. TTC (1.38) is **built and tested but not named**: S1Z
showed that a safety pair term on P6 bought fewer failures by **picking slower plans** (EP −0.449 separated adverse). Adding TTC
pairs before EP is repaired risks repeating it. TTC joins under O1 (`PREREG_SCORER_FT_DRAFT.md` §7).

---

## 3 · Launch-gate entries (CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS")

### 3.1 Profiles (`launch_gate.py`; new dicts, no existing profile edited)
`refe_scorerft_t` (T, seed 0), `refe_scorerft_c` (C, seed 0), `refe_scorerft_t_s1` (Ts), `refe_scorerft_c_s1` (Cs), all built by
`scorerft_profile(arm)` from `PROFILES["refe_p5c"]`. Each carries:
* `scorerft_rules: True` → `scorerft_refusals(prof, argv)`. The argv's flag SET and every non-path lever must equal the template
  `scorerft_argv(arm)`. **argv(T) − argv(C) = {`--rank-w` value, `--rank-heads`, `--rank-ep-gap`, `--rank-gate-topk`, `--out`}.** C
  names `--rank-w 0.0` explicitly (a default-off lever named in the arm's argv, CLAUDE.md "check the DEFAULT"). argv(Ts) − argv(T) =
  {`--seed`, `--out`}.
* Pins: `init_from_sha256 = SCORERFT_BASE_SHA256` (C1 final; **unfilled ⇒ REFUSE**), C1's table `traj_label_sha256 = 3b198314…` +
  `rank_policy rank0`, `expect_schedule` = {epochs 0, steps N, P7 bank `train_grow_cmd_3c_p7` by count 150,603 AND content
  `92af8d66…`}, a fresh `--out` (no `ckpt_last.pt`), and `rank_w` registered equal to the weight-rule output (float type checked).
* `live_terms = P6_TERMS + ("rank_L",)` **only on T/Ts.** ⚠️ On the field arms, a control that declared its term FAILED G-LIVE
  ("field_L ABSENT", C, 2026-11-19).
* `forbidden_flags` += `REFE_RANK_MUTATE` in the environment (the gate refuses a launch env that sets it).
* Suite += `test_rank_loss.py --fast` (`ZZRANK_PASS`) and `test_scorerft_profile.py --fast` (`ZZSCORERFTPROF_PASS`).

⚠️ **Two existing gate sites must learn the new key, or the control is mis-judged.** (a) `gate_checks.py:2348` routes only
`p5c_rules`/`p7field_rules` to `p5c_live_rows`, so it must also route `scorerft_rules`. (b) `p5c_live_rows` (h) (`gate_checks.py:2231-2235`)
requires a label table only when `p5c_arm == "C1"`, and FAILS any other arm that carries one. **All four SCORER-FT arms carry C1's
table**, so (h) must key on a profile field (`traj_label_required: True`), not on the arm name.

### 3.2 G-DVB rows (`DVB_KINDS` += `rank_w`, `rank_heads`, `rank_ep_gap`, `rank_gate_topk`: "built"; rows in `rank_dvb_rows`, the `field_dvb_rows` pattern `gate_checks.py:548-600`)
| lever | built side observed (never the namespace) | ok when |
|---|---|---|
| `rank_w` | a hook on `L_rank`'s output reads d(total)/d(L_rank) × accum on every labelled micro-batch of the SMOKE | == `rank_w` on **every** labelled micro-batch (C: the hook **never fires**) |
| `rank_w` | the logged `rank_w` on every row | single value == argv |
| `rank_heads` | the `rank` start event's built heads | == argv set |
| `rank_ep_gap`, `rank_gate_topk` | the built rule in the `rank` event, and the kwargs the term was called with (SMOKE capture) | == argv |
| `rank_*` (C) | no `rank` event, no `rank_L` on any row | absent |

### 3.3 G-LIVE rows (pod real-data smoke; `rank_live_rows`)
* **L1** `rank_L` finite on every logged row with a labelled set (T); absent (C).
* **L2** pair counts from the weight-rule JSON (32 real sets) plus the smoke: Σ EP pairs > 0 **and** Σ gate pairs > 0. A head with
  zero pairs is structurally dead, the "built, tested, unreachable" class.
* **L3 gradient reach, with zero controls** (one labelled micro-batch, `rank_w·L_rank` alone backward on the smoke's real model):
  non-zero on `score_head.weight` rows NC, DAC, EP and on `score_dec`, `score_q_mlp`, the trunk group (LoRA + `scene_proj`);
  **EXACTLY 0.0** on `score_head` rows TTC, C, DDC, and on the trajectory decoder / `traj_head`.
* **L4 pair audit:** the gate re-derives every EP/gate pair and `L_rank` from the captured (labels, logits) with its own numpy
  reference. Pair sets identical; `L_rank` within 1e-6 relative.
* **L5 harness identity on navtrain (closes the §1.5 UNVERIFIED):** for 8 smoke tokens, the labeller's 6 labels for proposal 0 ==
  NAVSIM's own `pdm_score` on the same pose array and parts, to 1e-9.
* (f) warm-start identity vs `SCORERFT_BASE_SHA256`; (h) C1's table present on all four arms.

G-HYG (new dests consumed, identity args), G-CLOCK (unchanged: `L_rank` consumes the lag-0 labels asserted at
`train.py:1915-1916`), G-EVAL (no model change, so the loader is unchanged), G-CKPT, G-SUITE (0 regressions), the post-gate G-ADAM
(`p5c_postgate.py`): unchanged, all run. ⚠️ Memory "pin edit moves profile fingerprints": run all launch-gate tests after adding
the profiles.

### 3.4 Deliberate-regression arms (each must turn the gate RED)
| arm | mutation | must fail |
|---|---|---|
| R-DROP | `REFE_RANK_MUTATE=drop`: `L_rank` computed, not added to `tot_` | G-DVB `rank_w` hook row; L3 (EP row reads 0) |
| R-UNSAFE | `=unsafe_ep`: EP pairs admit NC·DAC < 1 | L4 pair audit |
| R-SHUF | `=shuffle`: labels permuted within the set before pairing | L4 |
| R-LEAK | `=traj_leak`: `L_rank` taken on a `detach_scorer=False` forward (the existing regression path, `model.py:711-712`) | L3 `traj_head` zero control |
| R-HEAD | `=head_swap`: gate z reads the TTC logit instead of DAC | L3 TTC-row zero control |
| R-CTRL | profile: C declares `rank_L` live, or T's argv drops `--rank-w` | G-LIVE L1 (absent) / `scorerft_refusals` |

**Analytic identities (G-SUITE, `test_rank_loss.py`):**
* `topk = 64` ⇒ bit-identical to the unrestricted reference (a bar below everything);
* a unit-level `gap = 1.5` ⇒ zero pairs, `L_rank` exactly 0.0, gradient exactly 0.0 (a bar above everything);
* `gap = 0.02` ⇒ partial;
* accumulation: B 2 × accum 2 == B 4 × accum 1 bit for bit;
* `--rank-w 0` ⇒ losses, gradients and parameters bit-identical to the pre-change trainer (synthetic backend, tiny config).

---

## 4 · What is NOT changed
The model, the labeller, the label table, the bank (parity: P7's reserve-free bank, pinned by count and content), the planner rule
(`navsim_v1`), the eval path, the nav-command input. No input is added at inference. The term uses labels at training time only,
so the vision-only and goal-disjointness rules are unaffected.

## 5 · Cost (ESTIMATED; step time INHERITED from the MM brief, ~51 s/step at the P5-C recipe)
Cross-check: the field pair F+C (2 × 2,000 steps) was approved at ~57 L40-h (MEMORY, PI go 11-17) = 2 × 2,000 × 51 s.
`PREREG_P5C_NEXT.md` §5 B quotes ~27 L40-h per 2,000-step arm. `L_rank` adds about 4 × 64² pair operations per micro-batch and
**no labeller jobs**, so the step time is ESTIMATED unchanged.

| item | L40-h (ESTIMATED) |
|---|---|
| weight rule + G-LIVE smoke + launch gate, per profile | ~0.5 each |
| Stage 1: T + C, 1,000 steps each | **2 × 14.2 = 28.3** |
| Stage 2 (only if Stage 1 passes P1 and G1): Ts + Cs | **28.3** |
| **total (1,000 steps)** | **≈ 58–59** |
| same at 2,000 steps (C1 parity) | ≈ 115 |

Reads per arm (dev box, not L40): E3 logit pass, harness, the 64-proposal table (`p5c_prop_score.py`, CPU). Cost as P5-C's read
(**UNVERIFIED**; the MM fills it from the P5-C read logs). The L40 is busy with the field chain C → F → Cs → Fs. Its Cs − C will give
the P5-C-recipe training floor on the P7 base, which is informative but **not importable** as this rig's floor (a different base, table
and length).

## 6 · Build list for the implementer (not done here)
1. `PK/refe/rank_loss.py`: `RankLoss(heads, ep_gap, gate_topk)` with `.term(score, labels, msk, batch)` and `.facts()`, plus the
   mutation hooks.
2. `train.py`: the flags, refusals, identity args, the term at `:1919-1943` inside the online branch only, the `rank` event, and the
   log fields.
3. `weight_rule_rank.py` (from `weight_rule.py`).
4. `gate_checks.py`: `rank_dvb_rows`, `rank_live_rows`, the routing at `:2348`, and the generalised (h).
5. `launch_gate.py`: `scorerft_argv`, `scorerft_profile`, `scorerft_refusals`, and the 4 profiles.
6. Tests: `test_rank_loss.py`, `test_scorerft_profile.py`.
7. The pod chain (T → C → [gate] → Ts → Cs) with post-gates, from the field chain's template.

## 7 · Deliverable manifest
| artifact | lives in | note |
|---|---|---|
| `DESIGN_SCORER_FT.md` | `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/` only | this file; not staged (outside the repo; the MM lands it) |
| `PREREG_SCORER_FT_DRAFT.md` | same folder only | DRAFT; the MM freezes it by sha256 |
| scratch exports of `e17d040e` sources | session scratchpad | re-creatable from git; deleted at hand-back |
