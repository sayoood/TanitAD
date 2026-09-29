# REF-F prior art, theory and backbones — Stream F-R (2026-09-29)

**Status:** COMPLETE — staged in the index, **not committed**. Scope: transfer of **CLM-8B**
(Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré, Mirhoseini; Stanford + NVIDIA; released 2026-09-23;
Apache-2.0) to driving as reference arm **REF-F** (camera + ego state in, driving command out).
**TanitAD numbers** are quoted only from `Project Steering/MODEL_REGISTRY.md` (cited by section), the 2026-09-25 programme review
(tagged `INHERITED`) or the orchestrator brief (tagged `INHERITED (brief)`); the registry wins on any conflict.

**Evidence classes:** `PUBLISHED-PRIMARY` (paper/README/code/figure I read myself) ·
`PUBLISHED-SECONDARY` (press/vendor/third-party summary — one hop, re-verify before spending a GPU-day) ·
`PUBLISHED-RECALLED` (id/title verified this session; the *statement* is from memory of the paper — re-verify before quoting) ·
`INHERITED` (another repo doc, not re-verified) · `MEASURED (ours)` (computed here; script named) ·
`ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

**Headline findings** (evidence tags in the sections they point to)
1. The closest published design is **CoVer-VLA (arXiv:2602.12281, same authors)** — frozen SigLIP 2 + trainable patch pooling + a *numeric* action tower — **not** CLM-8B's frozen Qwen3-8B text encoder (§2.1, §6 #1).
2. **CLM is a chooser/verifier over given candidates, not a generator or controller:** zero-shot results are discrete choices among small candidate sets; its SOTA claims are best-of-N verification on n = 38 / 30 tasks; its only control-like demo (T-Rex) is label-reading with **≤ 29.2 %** shield interventions (§1.4).
3. **InfoNCE's argmax is PMI, not likelihood:** with in-batch negatives it ranks by evidence against the action prior; CLM's code has no correction; a toy shows the flip is real but conditional (§3.1, E-F1).
4. **Registry ceiling:** learned re-scorers recovered ≤ 8.4 % of REF-C's oracle gap (47 arms) and hard-argmin (= one-hot InfoNCE) is the worst target ⇒ REF-F needs soft listwise targets and a Δ2-paired claim (§6 #3-4).
5. **CLM's 0.6 ms needs a repeated state**; a driving tick is a new state (measured cold path ≈ 28 ms on an RTX 4090 for an 8B text encoder). **No Thor latency exists in what I could find for any candidate backbone** (§1.3, §4.2).
6. **Not read:** the CLM blog (scaling exponents, ablations) and the HF card — egress-blocked; nothing from them is quoted (§1.1, §6 #14).

---

## 1. CLM itself — beyond the brief

### 1.1 What was actually read (access log, so the reader can weight it)

| source | how | status |
|---|---|---|
| `github.com/Contrastive-LM/CLM` @ `bb42c6c` (2026-09-24) | **shallow `git clone`** (github.com is egress-permitted; done instead of WebFetch so code is read verbatim, not via a summariser). README, `docs/FINETUNING.md`, `train/finetune.py`, `src/clm/{heads,engine,schema}.py`, `evaluation/bon_eval.py`, `examples/t_rex/**`, both result JSONs, both figure PNGs (viewed) | PUBLISHED-PRIMARY |
| Blog `contrastive-lm.notion.site` (holds scaling fits, ablation tables, paper figures) | WebFetch → **EGRESS_BLOCKED** | **NOT READ** |
| `huggingface.co/Contrastive-LM/CLM-v0.1-8B` model card, weights | WebFetch blocked by policy; search-snippet only | **NOT READ** (checkpoint config unverified) |
| VentureBeat / AlphaSignal / Sanity glossary / MindStudio / DataCamp / LangChain / krino#76 | WebSearch summaries + 1 WebFetch (krino issue) | PUBLISHED-SECONDARY |
| README says *"The scaling experiments, data pipelines and paper figures live in the research repo's `main` branch"* | — | **not public in this branch** |

⇒ Everything below labelled PUBLISHED-PRIMARY is code/README/figure; **anything that lives only in the blog
(scaling exponents α_X, bi-vs-cross-encoder ablation, frozen-vs-fine-tuned backbone, InfoNCE-vs-Choice numbers,
failure cases) is UNVERIFIED and is listed in 1.6 as a gap, not guessed.**

### 1.2 Confirmed from code (PUBLISHED-PRIMARY) — the training/serving recipe REF-F would inherit

| item | value (file) |
|---|---|
| score | `exp(logit_scale)·cos(z_s, z_a)`, `logit_scale` init `ln(1/0.07)` (=14.3×), **clamped ≤ 100** (`heads.py`, `finetune.py`) |
| head | `4096 → width → … → 512`, GELU, LayerNorm, no residual; **default `width 1536, depth 3`** (`finetune.py`) |
| optimiser (`--task clm`) | AdamW, wd 0, **lr = 2e-3·√(1024/width)·√(batch/1024)**, OneCycle (pct_start 0.1, cosine), grad-clip 1.0, **batch 2048, 20 epochs**, early-stop patience 5 |
| loss | bidirectional in-batch InfoNCE; `_clm_loss` masks `(i,j)` with **equal `(task_id, step_idx)`** to −∞ (see 1.3) |
| split discipline | **task-disjoint** train/val/held-out; selection metric default `within_task_top1` = rank of the gold action among the *same task's other steps'* actions (≤4096 members) |
| sampler | `random` or **`task-blocked`** (`--tasks-per-batch 4`): batches built from few tasks ⇒ *hard, temporally-correlated in-batch negatives* |
| `--task choice` | `--loss infonce` (**default**: bidirectional in-batch over the batch's *distinct option texts*, so other questions' options are negatives) **or** `softce` (softmax over the question's own options); `--targets soft` (**default**, annotator distribution) or `hard` |
| inference | softmax over `scale·cos/temperature` of a question's options **is** the answer distribution; `confidence` = top prob − mean(others) (`schema.py`) |
| serving | encoder = vLLM Qwen3-8B pooling server, `--max-model-len 2048`, texts truncated via vLLM `truncate_prompt_tokens=2048` (training recipe `embed_utils.Recipe`: states keep the **tail**, actions the **head**); heads 512-d; LRU "vector arena" 2 % of device memory keyed by head generation |
| `clm-raw` | shipped **ablation**: cosine in the raw encoder space, scale fixed 100, no head (numbers are in the blog only) |
| trajectory verifier (`bon_eval.py`) | per-step cosine score → **mean over final `--window 12` steps** → pick argmax per task; reports **selected / random / oracle** with exact expectation over uniform N-subsets and uniform tie-breaking |

### 1.3 Corrections / refinements to the brief (each checked against source)

1. **False-negative mask is per *state step*, not per task.** `codes` groups rows by `(task_id, step_idx)`;
   only *other rollouts at the same step of the same task* are masked. Everything else from the same task
   stays a negative (and `--sampler task-blocked` builds batches from 4 tasks so that these same-task negatives dominate). *(finetune.py `_load_clm_data`, `_clm_loss`.)*
2. **Choice mode default is InfoNCE, not softmax-CE.** `softce` exists as the alternative; which one trained
   the released `clm-latest` head is **UNVERIFIED** (the released head comes from the 3-stage pipeline, not `--task choice`).
3. **"cold 58.1 ms" is one README example, not the latency table.** The measured table (RTX 4090, server p50,
   fixed action set) is **28.0–28.8 ms for a new state every call** and **0.6–0.7 ms for a revisited state**;
   the T-Rex run shows `model_ms_p50` **2.6 ms** and client-side p50 **16.5 ms**. The 0.6 ms figure needs a
   *repeated* state. **A driving loop never revisits a state (a new frame every 100 ms)** ⇒ the state side pays the
   full encoder every tick; only the action side (fixed vocabulary/anchors) caches. (ESTIMATED consequence.)
4. **Trainable-parameter count is ambiguous.** README: "20M-parameter head" per encoder. Default cfg
   (width 1536, depth 3, LN) = **9,443,840 params/head, 18,887,680 for the pair = 75.55 MB fp32** —
   matching the README's "75 MB reference head" to within rounding. Either the "20M" is the **pair**, or the released cfg is wider and
   stored in fp16. `MEASURED (ours, arithmetic on published cfg + published size)`; **UNVERIFIED** which (checkpoint
   not downloadable here). Immaterial to the design (≈0.2 % of 8B) but do not quote "2 × 20M" as fact.
5. **The N\* ∝ D^1.02 (~310 tokens/param) law is not admissible under TanitAD's exponent rule** — the README
   gives no fit window, R² or n (they live in the blog). Fit on Nemotron-DQA **text QA**; transfer to driving =
   HYPOTHESIS. Same for "encoder size gives the strongest gain": direction only.

### 1.4 Results re-read from the primary figures (`assets/*.png`, `examples/t_rex/results/*.json`)

`PUBLISHED-PRIMARY`; ratios/intervals are `MEASURED (ours)` arithmetic on published counts.

| benchmark | CLM-8B | Jev | note |
|---|---|---|---|
| T-Rex (5 seeds × 60 s, **shield on**) | 5/5 survive, latency p50 **16.5 ms** | 5/5, **149.8 ms** | ratio **9.08×** = the "up to 9×" |
| BFCL v4 tool calling | **95.2 %**, 76.8 ms | **99.2 %**, 125.5 ms | CLM **−4.0 pts**; speed-up 1.63× |
| WikiRacing | **26/30**, 79.8 ms | **30/30**, 225 ms | CLM Wilson-95 % [0.703, 0.947] vs Jev [0.886, 1.000]; 2.82× |
| Super Mario (5 seeds) | 5/5, 33.5 ms | 5/5, 132.6 ms | 3.96× |
| DeepSWE verifier (Bo4, **38** held-out tasks) | **81.6 %** (31/38; Wilson-95 % **[0.666, 0.908]**) | 71.1 % | dashed **pass@1 73.7 %** ⇒ CLM **+7.9 pts** over random pick; 79 vs 449 ms = 5.68× |
| Terminal-Bench 2.1 verifier (Bo5, **30** tasks) | **87.6 %** | 83.1 % | pass@1 **84.0 %** ⇒ CLM **+3.6 pts**; 32 vs 131 ms = 4.09× |

* "On par with Jev" = **tied only where n = 5 seeds** (T-Rex, Mario); **behind on both benchmarks with resolution**
  (BFCL −4.0 pts, WikiRacing −13.3 pts). The speed-up ranges **1.6×–9.1×**; **9× is the single best case** (tiny cached
  action set). Median of the four zero-shot ratios = 3.4×.
* **Third-party misreading caught:** krino#76 lists "DeepSWE CLM 73.7 % / Terminal-Bench 84.0 %" — those are the
  **pass@1 dashed baselines**, not CLM. The primary figure agrees with the README (81.6 / 87.6). Do not cite krino's table.
* The verifier gains are measured against **pass@1 (random pick)**; **no oracle (best-of-N) bar is printed**, and the
  headline denominators include all-pass/all-fail tasks that no selector can get wrong (`bon_eval.py` computes
  `mixed_tasks` but the README does not print it). This is exactly TanitAD's oracle-gap framing — **CLM's harness computes the oracle (`oracle_any`)
  but the README and figures publish only selector-vs-random.**

**T-Rex is not evidence of control skill (read the harness before quoting it).** `examples/t_rex/README.md`:
a hand-written physics planner labels each action *safe/unsafe/best* **inside the candidate texts** ("jump: Safe.
Clears the 2 large cacti. Best."), so the model reads the answer; a **shield** replaces unsafe answers and an
emergency check can act first. From the per-seed rows: CLM **agreement with the planner's best move 65.8 %**
(Jev **98.7 %**); interventions (shield + arrival + emergency) **4,883 / 16,709 decisions = 29.2 %** (Jev
**28 / 5,595 = 0.5 %**). Both survive because of the shield. `MEASURED (ours)` from the published JSON
(`summary.shield_interventions` = Σ shield 364 + Σ arrival-saves 4,519 + emergency 0 — three harness counters, each meaning the executed action differed from the model's answer
(`pilot.py`: `executed != proposed / original`); they could overlap, so read 29.2 % as an **upper bound**). ⇒ the *only* closed-loop use in the repo is a
label-reading task in which the model needed intervention on ~3 of 10 decisions.

### 1.5 Verified absences in the released code (grep, commit `bb42c6c`) — `MEASURED (ours)`

`grep -rniE 'calibrat|ECE|brier|conformal|prior|marginal|debias|false.neg|logq'` over `*.py`/`*.md`: **no** calibration
loss or metric, **no** log-prior / log-Q correction, **no** debiased-contrastive term; the *only* false-negative device
is the same-step mask (1.3-1). The softmax over options is presented as the answer distribution with **no calibration
evidence in the repo**. (A third-party reading, krino#76, says the same; PUBLISHED-SECONDARY.)

### 1.6 Open items about CLM (what the blog would have to answer) — all `UNVERIFIED`

| question | status |
|---|---|
| bi-encoder vs cross-encoder **ablation** | none in README/code. Search summaries give only the *design argument* ("13× faster at ~1k candidates, no fine-grained state–option interaction" — krino author's own analysis, PUBLISHED-SECONDARY). **No published head-to-head found.** |
| frozen vs fine-tuned backbone | backbone is frozen by construction; a fine-tuned-backbone arm is **not reported** in what I could read |
| InfoNCE vs Choice-softmax numbers | not in README; blog only |
| scaling exponents α_C, α_D, α_N, α_Nenc | blog only; only the ordering "encoder size strongest" and N\*∝D^1.02 reach the README |
| failure cases | none published beyond "Jev fails as a long-horizon verifier"; BFCL/WikiRacing deficits above are the visible ones |
| hard-negative recipe | PUBLISHED-PRIMARY: pretrain-only **52.1 %** → +mid-train **69.2 %** top-1 on ~100 K held-out questions (1 gold + 10 hard negatives); hard negatives **from the start peak at 62.4 %** then overfit (⇒ 7 pts worse); post-training with 40 % Nemotron replay keeps 69 % → **68.5 %**, **agentic-only drops to 56.2 %** |

### 1.7 Jev, the multimodal CLM, robotics/driving status

* **Jev** = TypeSafe AI's *System One* model (launched 2026-09-15 per press): **non-autoregressive**, emits its whole
  typed output (`noul`/`choice`/`score`) in a **single parallel pass**, trained with "Reinforcement Learning for
  Calibrated Decisions (RLCD)"; latency 70–500 ms; vendor claims 20–200× faster / 40–400× cheaper than LLMs.
  `PUBLISHED-SECONDARY` (MindStudio, DataCamp, LangChain, Analytics India — search summaries; vendor claims). CLM's API
  is deliberately **TypeSafe wire-compatible**. Note the contrast: Jev is *trained for calibration* (RLCD); CLM is not.
* **Release date:** the engine's `RELEASE` constant and the `/v1/models` example say **2026-09-19** (head date); public release 2026-09-23 per the brief and press.
* **CLM-35B-A3B** (multimodal MoE): "early October 2026" per two secondary sources; the README roadmap says only
  *"Vision and multimodal support: images, video and other modalities for robotics and computer-use tasks."*
  **As of 2026-09-29 it is announced, not released** — `PUBLISHED-SECONDARY`, UNVERIFIED against the blog.
* **Robotics / driving:** none published. Marco Pavone (Stanford / NVIDIA Research) is a co-author and the roadmap names
  robotics, but two targeted searches and the repo contain **no driving or robot result**. REF-F would be the first
  public transfer — i.e. there is **no external number to calibrate expectations against**.

---

## 2. Prior art REF-F must position against

Tags: `PUBLISHED-PRIMARY` = I read code/README/figure. `PUBLISHED-SECONDARY` = search-engine summary of the paper/page (arXiv
itself is egress-blocked here; **every arXiv-sourced number below is one hop unless it says PRIMARY**). `INHERITED-repo` = read from a
prior TanitAD doc that read the paper (its tier in brackets; `…/Research/2026-07-27-planner-scorer-inputs/CITATIONS.md` unless stated);
arXiv ids were re-confirmed by search this session or by the 2026-09-25 V1 citation check (`Project Steering/Reviews/…/V1_citation_check.md`).

### 2.1 Nearest neighbours: contrastive state–action **verifiers** (robotics, driving)

| system | what it shows | implies for REF-F |
|---|---|---|
| **CoVer-VLA** — Kwok, Zhang, Xu, Liu, Mirhoseini, Finn, Pavone, *"Scaling Verification Can Be More Effective than Scaling Policy Learning for VLA Alignment"*, **arXiv:2602.12281** (Feb 2026; Stanford+NVIDIA; **same first author as CLM; Mirhoseini and Pavone on both**). `PUBLISHED-PRIMARY` (code `cover-vla/cover-vla @21a9960` + README) | **Architecture (read from `bridge_verifier/ensemble_eval/finetune_trajectory_bridge_ddp.py`, `model.py`):** *frozen* SigLIP 2 (open_clip, bf16) + trained ClearCLIP-style text-aware patch attention + 4-layer attention-pooling towers; **action tower = numeric action-chunk encoder trained from scratch** (MLP over a flattened 20×7 history, or 4-layer Transformer + masked-mean); L2-norm; `logit_scale` init ln(1/0.07); **symmetric in-batch InfoNCE** (`cross_entropy(…, arange(B))` both ways); *no negative mining or false-negative handling found (grep for mask/duplicate/hard/negative hits only padding masks and dataset statistics); plain `DistributedSampler` shuffling*; **ensemble of verifiers** on one frozen backbone at inference; checkpoint ~312 MB. **Results (README, SIMPLER, 100 trials/task, 8 rephrases × 5 action samples = 40 candidates):** ID avg **41.5 → 57.0** (π0+CoVer) **→ 65.5** (rephrase+CoVer); OOD avg **29.7 → 61.0 / 62.0**; one ID task unchanged (Carrot-on-Plate 48 → 48); README headline "+22 % ID / +13 % OOD over scaling policy learning alone" (baseline definition **not in README → UNVERIFIED**); PolaRiS (π0.5) task-progress **40.0 → 53.9**, success **3.8 → 13.1 %** (table is commented out in the README). README to-do: *"Release verifier training pipeline"* still unchecked | **This, not CLM-8B, is the design REF-F should copy**: a *numeric* action tower (not an 8B text LLM), a frozen vision backbone with **trainable pooling over patch tokens** (not one frozen last-token vector), and a verifier scoring a *diverse generated candidate pool*. The paper's thesis (search summary, SECONDARY): jointly scaling rephrases × action samples raises test-time diversity and recovers correct actions more efficiently than scaling either alone — REF-C's fan is already diverse (oracle 0.1640, `MODEL_REGISTRY.md` §4.1). Caveat: robot manipulation, success-rate metric, n=100/task; **no closed-loop driving evidence anywhere** |
| **VLA-R** — *"Vision-Language Action Retrieval toward Open-World E2E Autonomous Driving"*, **arXiv:2511.12405** (Nov 2025). `PUBLISHED-SECONDARY` | frozen open-world VLM perception + Q-Former bottleneck + **vision–action contrastive alignment**, action *retrieval*; authors state the study is limited to a **small dataset (< 2 h of driving)**; no numbers read | only driving-domain contrastive vision–action paper found; **too small to calibrate expectations**; note the same recipe (frozen VLM + Q-Former + InfoNCE) |
| RoVer (**2510.10975**), Action Draft-and-Verify (**2603.18091**), VeriSpace (**2606.10568**), VerNav (**2609.00920**), CLASP *"Contrastive Language, Action, and State Pre-training for Robot Learning"* (OpenReview `sxKR6zhBDH`), TraVEL (**2608.13495**, InfoNCE for driving-*video* retrieval, not planning). `PUBLISHED-SECONDARY`, **titles/ids seen in result lists only — content NOT read** | a 2025–26 cluster of test-time action verifiers for VLAs; RoVer = compact process-reward model; ADV = one VLM pass scores all candidate chunks by length-normalised NLL | the verifier-over-candidates pattern is now standard in robotics; **no driving-scale dual-encoder trajectory scorer found** in 3 query phrasings + the prior-art tables of the repo's scorer-inputs survey (verified absence at limited depth) |

### 2.2 Implicit Behavioral Cloning and Diffusion Policy

| system | what it shows | implies |
|---|---|---|
| **IBC** — Florence et al., **arXiv:2109.00137** (CoRL 2021). `PUBLISHED-PRIMARY` (code `google-research/ibc @db89ddb`, 2024-01-25) + `SECONDARY` (abstract) | policy = energy model, action = argmin energy at inference. **Loss (read from `ibc/losses/ebm_loss.py`, `ibc/agents/ibc_agent.py`):** `info_nce` = softmax cross-entropy over `[B × (n+1)]` with the true action in the last column (“can you classify the correct example?”); **n = 256 counter-examples per state (default), drawn uniformly from the action-spec bounds** (`sample_spec_nest`) and, by default (`fraction_langevin_samples=1.0`), refined by Langevin MCMC; `softmax_temperature=1.0`; **`add_grad_penalty=True`**. Variants shipped in the same file: `cd`, `cd_kl` (code comment cites arXiv 2012.01316), **`clipped_cd` / `soft_clipped_cd` — counter-examples within a squared-error threshold (0.2) of the true action are masked or softly shaped**. Abstract: implicit policies *often outperform* explicit MSE/mixture-density BC, incl. high-dim actions and image inputs; competitive with offline RL on D4RL human-expert tasks | the **direct ancestor of REF-F's loss**, with two lessons the CLM line dropped: (i) IBC's negatives are **per-state uniform (+ model-sampled) actions, not other rows' true actions** ⇒ by 3.1 there is **no data-marginal prior term** (uniform ⇒ constant log q; MCMC samples ⇒ MLE-like), whereas CLM/CoVer in-batch negatives carry it; (ii) IBC's own authors added a **near-duplicate mask** (`clipped_cd`) — precedent for “any candidate within δ of the GT is not a negative” (3.4). Note the gradient-penalty regulariser: energy-scale control was needed for stability |
| **Diffusion Policy** — Chi et al., **arXiv:2303.04137** (RSS 2023). `PUBLISHED-SECONDARY` (summaries of §4 text) | avg **+46.9 %** over SOTA baselines (BET, LSTM-GMM, IBC); on IBC: *"training error spikes repeatedly and success oscillates, making checkpoint selection difficult"*; *IBC fails to infer training actions with increasing accuracy despite smoothly decreasing training loss*; IBC/LSTM-GMM biased to one mode | (i) **monitor held-out argmax accuracy, never the InfoNCE loss** (CLM's own `within_task_top1` selection metric is the right habit); (ii) IBC's instability stems from **continuous-action argmin + negative sampling**; REF-F scores a *finite candidate set*, removing the first but not the second; (iii) checkpoint selection must be pre-registered |

### 2.3 Contrastive RL and contrastive critics

| system | what it shows | implies |
|---|---|---|
| **Contrastive RL** — Eysenbach et al., *"Contrastive Learning as Goal-Conditioned RL"*, **arXiv:2206.07568** (NeurIPS 2022). `PUBLISHED-SECONDARY` | critic = **inner product** φ(s,a)ᵀψ(g) trained with InfoNCE; the inner product *is* a goal-conditioned value function; beats non-contrastive baselines incl. offline and image-based, without augmentation | ⭐ **contrast is over *goals*, not actions**: for fixed (s,g) the p(g) term is action-independent, so `argmax_a` **is not prior-biased**. CLM/CoVer contrast over *actions* and are. REF-F variant: contrast (s,a) against the realised *future latent* (HYPOTHESIS) — subject to the binding goal-admissibility rule (goal from a predicted geometric point, never from the situation classifier) |
| **1000-Layer Networks for Self-Supervised RL** — Wang, Javali, Bortkiewicz, Trzciński, Eysenbach, **arXiv:2503.14858**. `PUBLISHED-SECONDARY` | depth up to **1024 layers** lifts contrastive-RL performance **2×–50×** (unsupervised goal-conditioned) | CLM's heads are 3-layer MLPs on frozen features; contrastive critics may be **depth-starved**, not width/encoder-starved (HYPOTHESIS; not tested in driving) |
| Contrastive Difference Predictive Coding (**arXiv:2310.20141**, title from result list only) | TD-style InfoNCE for future-state prediction | bootstrapped variant if a multi-step value is wanted; content not read |

### 2.4 Vocabulary / probabilistic / scored planning in driving (the incumbents)

`INHERITED-repo` throughout (PDF-VERBATIM unless noted); ids CONFIRMED by V1 where listed there.

| system | what it shows | implies |
|---|---|---|
| **VADv2** (2402.13243) | 4,096-entry vocabulary by furthest-point sampling of demonstrations; `p(a∣s)` = softmax over the vocabulary; `L_distribution` = KL to a **soft, distance-weighted** target (GT added as positive, nearby trajectories "less penalised"). **T.3 (1 s L2, CARLA): full 0.082 · −image tokens 0.083 · −map 0.086 · −agent 0.089 · −distribution loss 1.415 (17×).** No vocabulary-size ablation | **supervision shape ≫ inputs** (17× vs ~0.001 m). VADv2's softmax is a *conditional likelihood*, not a PMI |
| **Hydra-MDP** (2406.06978, V1 ✓) / **++** (2503.12820) | 4,096/8,192 vocabulary (k-means over ~700 K nuPlan trajectories); 5 rule teachers distilled **per candidate**: distilling the *scalar* PDM score **80.2 < imitation-only 80.9 < five separate heads 83.0**; ++: 85.0 → 86.5 PDMS, 76.8 → 80.6 EPDMS (matched vocabulary, target-only change). Inference = grid-searched weighted sum of per-head log-scores | **a single scalar score is the weaker design** ⇒ REF-F should expose **per-family sub-scores** (also mandated by the four-family eval rule) |
| **GTRS** (2506.06664, V1 ✓, NAVSIM v2 challenge winner) *(HTML-SUMM)* | scorer over a super-dense vocabulary: random selection **25.6 → 39.7 EPDMS** (GTRS-Dense); **+11.1 EPDMS zero-shot on unseen proposals**; navhard GTRS-E 49.4 vs privileged PDM-Closed 51.3 | generalising to **unseen candidates** is the dual-encoder's genuine selling point (shared action tower); a fixed-vocabulary classifier cannot |
| **DriveSuprim** (2506.06659, AAAI 2026, V1 ✓) | oracle over a 256-fan: **top-1 91.9 · top-4 94.5 · top-16 96.1 · top-256 98.7 PDMS vs human 94.8**; *"easy-to-reject options dominate the training process and gradient"*; fix = coarse-to-fine (8,192 → 256 → dedicated decoder) + soft self-distillation labels (δ=0.15): **89.9 → 93.5 PDMS** | in-batch random negatives are exactly "easy-to-reject"; retrieve-then-rerank is the published fix (3.3) |
| **PDM-Closed** (2306.07962) + **LLM-Assist** (2401.00125) | rule scorer over **15** IDM proposals (3 lateral × 5 speeds), weights EP 5 / TTC 5 / Comfort 2; Val14 CLS-R 92 vs log-replay 80. LLM-Assist T.1: **15 proposals 92.51 → 8,505 proposals 77.78 CLS-NR** (progress *rises* 91.75 → 95.60) | a *fixed* scorer degrades as the candidate set grows — TanitAD's own **K4** measurement (`Research/2026-08-03-refc-planner-vision-research/REFC_PLANNER_VISION_RESEARCH.md`, MEASURED there: oracle-in-fan 0.2213 / 0.1914 / 0.1640 for 64 / 128 / 256 anchors while `frac_sel_2x_worse` rises 0.3825 / 0.4109 / 0.4540) is the same phenomenon |
| **WoTE** (2504.01941, V1 ✓) · **CLOVER** (2605.15120) | WoTE: 256 anchors, per-candidate BEV world-model rollout + reward head (BCE on simulator NC/DAC/TTC/Comf/EP): **81.0 → 83.2 → 85.6 PDMS**. CLOVER: K=64 oracle 0.9933 vs selected 0.9369; **expanding proposals moved the oracle, not the pick** | consequence-aware scoring is what moves selection; a state-only cosine has no rollout |

### 2.5 Retrieval and nearest-neighbour policies

| system | what it shows | implies |
|---|---|---|
| **VINN** — Pari et al., *"The Surprising Effectiveness of Representation Learning for Visual Imitation"*, **arXiv:2112.01511** (RSS 2022). `PUBLISHED-SECONDARY` | decouple representation (self-supervised) from behaviour: **locally-weighted k-NN over frozen embeddings**, *no behaviour training*; beats parametric BC on offline demos and real-robot door opening | **the zero-training falsification baseline for REF-F's frozen state encoder**: if k-NN over the frozen embedding cannot beat hold-speed on non-steady windows, no head on top of it will (consistent with the frozen DINOv2 / I-JEPA metric failures, `INHERITED (brief)`) |
| **Behavior Retrieval** (Du, Nair, Sadigh, Finn, 2023) | *recalled only* — **not found in search; id and numbers UNVERIFIED, not used** | — |
| VLA-R, VLADriver-RAG (**2605.08133**, title only) | retrieval-augmented driving VLAs | retrieval of *past expert actions* by state similarity = REF-F's cached action arena used as a memory (HYPOTHESIS) |

### 2.6 System-1 / System-2 driving

| system | what it shows | implies |
|---|---|---|
| **DriveVLM-Dual** (**2402.12289**) `SECONDARY` | slow VLM + fast conventional pipeline (VAD); SOTA nuScenes planning when cooperating with VAD; real-time **asynchronous** operation on production hardware, **~410 ms average VLM inference** | System-2 runs at ~2 Hz; System-1 must not wait on it |
| **FASIONAD** (**2411.18013**) `SECONDARY` | fast/slow dual system with **uncertainty-gated** switching + a nuScenes-derived fast-vs-slow benchmark | the hand-off needs a **calibrated confidence**; CLM provides none (1.5) |
| ETA (**2506.07725**) `INHERITED-repo` (W1 stream) | "thinking ahead" fast/slow | corroborates the split; W1 already logs it as external validation of the programme's cadence hierarchy |
| **CLM / Jev** (language agents) | "System One" = small repeated typed decisions and **verification of a System-2 generator's candidates** (CLM: best-of-N verifier) | REF-F = System-1 scorer over REF-C's fan; the generator/verifier split is the same as CoVer and DriveVLM-Dual |

---

## 3. Theory to get right

### 3.1 What the InfoNCE critic estimates — and what `argmax` over candidates then does

**Result** (`PUBLISHED-RECALLED` for the attribution — ids verified by search, statements from memory of the papers, re-verify before quoting — **and** `MEASURED (ours, exact toy)`: §5.2 B3 verifies (1) numerically).
Draw the positive a⁺ ~ p(a|s) and K−1 negatives i.i.d. from a **proposal** q(a). The InfoNCE-optimal critic is

    f*(s,a) = log p(a|s) − log q(a) + c(s)                                            (1)

van den Oord et al. 2018 (**arXiv:1807.03748**) for q = the marginal p(a), where (1) is the pointwise mutual information; Poole et al.
2019 (**arXiv:1905.06922**); the general-q form is the NCE / ranking-NCE analysis of Ma & Collins, EMNLP 2018 (**ACL D18-1405**; arXiv id
**UNVERIFIED** — a search summary states the *ranking* variant is consistent under weaker assumptions than the classification variant).

**Bidirectional loss** (`MEASURED (ours, exact toy)` — §5.2 B3, not a citation): the s→a term pins f up to c(s) with proposal q_a, the a→s term up to
c′(a) with proposal q_s; both hold iff `f = log p(s,a) − log q_a(a) − log q_s(s) + k`. B3 minimises the *exact* population loss (K = 3 candidates, 3×4 discrete world, arbitrary proposals)
and recovers this to **4 d.p. (max deviation 0.0000)**; the fixed-state ranking then equals `argmax_a[log p(a|s) − log q_a(a)]` and differs from the MAP action in 2 of 3 states. **For ranking actions at one fixed state only q_a matters** —
the symmetric loss (CLM, CLIP, CoVer) does not remove the prior term.

Consequences:
1. **c(s) is unidentified ⇒ scores are comparable only within one state** — never across frames, scenes or ticks (no absolute confidence).
2. With **in-batch negatives** (CLM, CoVer) q_a = the data marginal, so `argmax_a f* = argmax_a [log p(a|s) − log p(a)]`: candidates are ranked by
   **evidence** (likelihood ratio vs prior), not posterior. Under a ~74 %-cruise prior, cruise actions carry large p(a) and are **penalised**;
   a rare manoeuvre a₂ beats the modal a₁ **iff p(a₂|s)/p(a₁|s) > p(a₂)/p(a₁)** — whenever the scene raises its odds by more than its prior odds.
3. **Toy check** (`MEASURED (ours, toy)`, code in §5.2; an illustration of (1), *not* a REF-F magnitude). K=64 actions, 4,000 scenes, 74 % steady
   (p=0.97 on a cruise anchor), 26 % ambiguous (0.60 cruise / 0.40 one rare manoeuvre); marginal cruise mass 0.874 by construction; exact critic f*:
   raw `argmax f*` matches the MAP action in **100 %** of steady scenes but **0 %** of ambiguous scenes (e.g. p(a₁|s)=0.60, p(a₁)=0.110 vs
   p(a₂|s)=0.40, p(a₂)=0.0018 — conditional odds 0.67 vs prior odds 0.016); E|err| **5.58** vs **3.95** for MAP (action-axis bins);
   `argmax(f* + log p)` recovers MAP exactly. **Negative control:** when every p(a|s) is a shifted copy of one Laplace kernel (so the marginal is a
   mixture of the same kernel and cannot decay faster than its fastest component), `argmax f*` = MAP in **4,000/4,000** scenes for λ ∈ {0.5, 0.8, 1.5, 3.0}.
   ⇒ **the bias is not automatic**: it needs multimodal conditionals whose secondary modes are globally rare — exactly "brake vs hold". Its size
   for REF-F is an empirical question (E-F1, 3.6). The CLM code has **no** correction of any kind (1.5).
4. **Corrections** (equivalent at the optimum, different in finite data): (a) post-hoc **logit adjustment** `f + α·log p̂(a)` (Menon et al.,
   **arXiv:2007.07314** `SECONDARY`: "large relative margin between logits of rare vs dominant labels", post-hoc or in-loss); (b) **training-time logQ
   correction** (Yi et al., RecSys 2019, doi 10.1145/3298689.3346996: subtract the log-probability of an item appearing in the batch; revisited in
   **arXiv:2507.09331** "Correcting the LogQ Correction" — title only, content unread); (c) **choose q with constant log q** — uniform over a fixed
   deployment set (IBC's per-state uniform counter-examples, `PRIMARY`; *hard negatives = the whole fan drawn uniformly*); (d) train a **listwise softmax over the deployment
   set** (CLM's `softce`, VADv2's KL), which estimates p(a|s) on that set directly. For continuous trajectories p̂(a) = soft anchor-cluster
   frequency or a small unconditional density head; tune α ∈ [0,1] on episode-disjoint validation.
5. **Ceiling:** the InfoNCE bound cannot exceed log K (van den Oord 2018; Poole 2019; RECALLED): K = 2,048 ⇒ **7.6 nats**. Random in-batch negatives
   are far from the positive, so *fine within-scene* ordering (0.16 m vs 0.47 m) is never supervised — DriveSuprim's "easy-to-reject options dominate
   the gradient" (2.4). Monitor **held-out argmax accuracy**, not the loss (IBC, 2.2).

### 3.2 Softmax over a fixed vocabulary = conditional likelihood — and when a dual encoder *is* that

VADv2 / Hydra-MDP / DiffusionDrive / REF-C's t=0 anchor classifier train `p(a_k|s) = softmax_k g_k(s)` with (soft or hard-argmin) cross-entropy **over the
deployment vocabulary**. The optimum is the true conditional *on that set*, `argmax` = MAP, log-loss is a proper scoring rule — **no prior term**.
A dual encoder over the *same fixed* K candidates is a **cosine classifier whose output layer is factorised, W = [v̂₁…v̂_K], v̂_k = action_tower(a_k)**;
with an action lookup-table it *is* a linear classifier on the 512-d state embedding. It differs only in: (i) **open-set / per-scene candidate sets**
(GTRS: +11.1 EPDMS on unseen proposals), (ii) **parameter sharing** across candidates, (iii) **cacheability** (the action arena), (iv) the **rank cap** (3.3).
⇒ *Which loss when:* stage 1 (representation, open vocabulary) = in-batch InfoNCE; stage 2 (deployment set = the fan) = **softmax-CE over that set with
soft distance-weighted targets** (CLM's own `choice --loss softce --targets soft`); stage-2 logits need no correction, stage-1 logits need `+ log q̂`.
`MODEL_REGISTRY.md` §4.1 (REF-C v1.2, MEASURED by TanitAD): **hard-argmin is the worst target in all five feature configurations (pointwise ≈
warm-listwise > cold-listwise > hard)** — and a one-hot InfoNCE positive *is* a hard-argmin target.

### 3.3 Dot-product (late-interaction) scoring vs cross-encoders; retrieve-then-rerank

* **Rank cap.** `s = ⟨u(s), v(a)⟩` realises only score matrices of sign-rank ≤ d. Weller et al. (**arXiv:2508.21038**, Google DeepMind/JHU, Aug 2025;
  `SECONDARY`): the number of distinct top-k subsets an embedding can return is bounded by d (sign-rank / communication complexity); a realistic
  dataset (LIMIT) with trivial queries defeats SOTA embedders **even when embeddings are optimised directly on the test set**.
* Driving costs are **interaction** terms (candidate geometry × the *specific* agent/obstacle positions: collision, TTC) — a high-rank pattern that a
  512-d bilinear score must memorise scene by scene. Cross-attention (candidate tokens attend to scene tokens) has no such cap.
* **Middle path — ColBERT** (Khattab & Zaharia, **arXiv:2004.12832** `SECONDARY`): both sides encoded independently, then a cheap MaxSim over
  *multi-vector* sets (abstract: BERT-level effectiveness, up to 170× faster than BERT-based rankers) ⇒ scene-token set × waypoint-token set.
* **Retrieve-then-rerank** is the published fix: DriveSuprim 8,192 → 256 → dedicated decoder (**+3.6 PDMS**); registry: top-K plateau **K = 8–32**.
  ⇒ dual encoder = stage 1 over a large *cached* vocabulary; a small cross-attention **expected-cost regressor** = stage 2 on the K = 8–32 survivors.
* CLM publishes **no** bi- vs cross-encoder ablation (1.6). The krino author's "13× faster at ~1 k candidates, no fine-grained interaction" is *analysis*,
  not measurement (`SECONDARY`). **E-F4 (3.6) is the missing ablation.**

### 3.4 False negatives, hard negatives, soft targets

* In-batch negatives assume different rows are different actions. Two random windows share a *functionally identical* action with probability
  ≈ Σ_k π_k² over action clusters at the eval tolerance δ; stopped/steady-speed windows form heavy clusters. **The true rate is a one-hour measurement**
  on the cached corpus (fraction of in-batch pairs whose GT trajectories are within δ ADE, δ ∈ {0.1, 0.2, 0.5} m) — **not estimated here** (the
  toy's 0.874 is by construction). These pairs get pushed apart although the "right answer" is the same ⇒ spurious repulsion.
* CLM's only defence is the (task, step) mask (1.3-1): it removes *duplicate rollouts of one state*, not *semantically duplicate actions of different
  states*. CoVer's script shows none.
* **Debiased contrastive learning** (Chuang et al., **arXiv:2007.00224** `SECONDARY`): correct the negative term for the probability that a "negative"
  shares the anchor's class — here "class" = action cluster. **Hard negatives** (Robinson et al., **arXiv:2010.04592** `SECONDARY`): sample negatives with
  controllable hardness — but hardness *raises* false-negative risk (near-duplicates of the GT are the hardest), the same tension as CLM's own finding:
  hard negatives from step 0 peak at **62.4 %** vs **69.2 %** for pretrain-then-refine (1.6).
* **Soft targets resolve both:** NCE with soft targets for conditional models (**arXiv:2404.14076**, title from result list, unread), CLM `--targets soft`,
  VADv2 distance-weighted KL, DriveSuprim δ = 0.15, registry v1.2 soft distance-weighted target. **Rule for REF-F: any candidate within δ of the GT, in the
  metric the eval uses, is a weighted positive, never a negative.**

### 3.5 Calibration of contrastive scores

* By (1), `softmax(scale·cos/T)` over candidates is **p(a|s)/q(a) renormalised** — a *tilted* posterior, calibrated only if q is uniform over the candidate
  set. CLM ships it as "the answer distribution" (`temperature` divides logits, `confidence` = top − mean(rest)) with **no calibration objective, metric or
  evidence in the repo (1.5)**; Jev, by contrast, is *trained* for calibrated decisions (RLCD, `SECONDARY`).
* **Temperature scaling** (Guo et al., **arXiv:1706.04599**: single scalar fit on held-out likelihood, "surprisingly effective"): necessary, not sufficient —
  it cannot repair a *prior* mismatch (3.1). Fit **after** the log-q correction, on **episode-disjoint** validation, and report ECE with the
  **episode-cluster bootstrap** CI (windows inside an episode are not exchangeable).
* **Conformal prediction** (Angelopoulos & Bates, **arXiv:2107.07511**): distribution-free coverage for a *set* of candidates ⇒ the **System-1 → System-2
  gate** (FASIONAD-style): if the conformal set contains candidates whose kinematics differ by more than a threshold, hand off to the slower planner or
  a hold-speed fallback. Caveat (HYPOTHESIS): exchangeability fails for time-correlated windows ⇒ split conformal by **episode**, not window.
* c(s) non-identifiability (3.1-1) means the raw cosine cannot be thresholded across frames; a gate needs the *normalised set distribution* or a conformal quantile.

### 3.6 Cheapest discriminating experiments — proposals for the PI; none run; both outcomes committed in advance

Common protocol: parity corpus `physicalai-train-e438721ae894`, paired episode-cluster bootstrap on the same windows, all four metric families, steady and
non-steady windows reported separately (74 % of windows are steady cruising; every arm loses to hold-current-speed there — `INHERITED (brief)`).

| id | question | arm A vs arm B | if A | if B |
|---|---|---|---|---|
| **E-F0** | is a frozen embedding metric at all? | VINN-style k-NN over frozen embeddings (+ v0, ax_fd) vs hold-current-speed, non-steady windows | k-NN beats hold-speed, CI ∌ 0 → backbone carries signal; proceed | k-NN ≤ hold-speed → frozen backbone is a dead end (as DINOv2/I-JEPA); go video-native/fine-tuned before any contrastive work |
| **E-F1** | does the prior term matter? | stage-1 logits with vs without `+α·log p̂(a)` | Δ (any family) CI ∌ 0 → adopt, log magnitude | no separation → drop it; record that the toy's bias did not materialise |
| **E-F2** | target shape | one-hot positive vs soft distance-weighted positives (same loss/data) | soft wins (registry + VADv2 predict) → adopt | one-hot ≥ soft → contradicts registry §4.1 and VADv2; escalate |
| **E-F3** | curriculum | in-batch only vs in-batch → fan hard negatives (soft) → replay (CLM recipe) | staged wins → adopt CLM curriculum | flat → hard negatives are not the lever on this corpus |
| **E-F4** | expressivity | cosine dual-encoder vs 2-layer cross-attention scorer, identical frozen features and targets | cross ≫ cos → dual encoder only as stage 1 | equal → keep the cheap cosine; rank cap not binding |

Working ids only. Against the orchestrator's pre-registration (`Project Steering/PREREG_REFF_CONTRASTIVE_SELECTOR.md`, plan §4.2): **E-F1 ≡ H-F4** (prior bias); **E-F4 ≡ the refuted branch of H-F1** ("the dot-product bottleneck binds");
**E-F0 is not in the plan** — a zero-training k-NN pre-gate for H-F7 that costs minutes; E-F2 / E-F3 are ablations inside plan §3.2–3.3, not separate hypotheses.

---

## 4. Frozen or foundation vision backbones for REF-F's state encoder

`PUBLISHED-PRIMARY` = model table read in the official repo I cloned (commit in the access log, §5.1). **No published Jetson Thor/Orin latency was found for
any candidate below** (4.2). "Video-native" = trained on and consuming spatio-temporal input.

| backbone | sizes (params @ input res) | video-native? | what is notable for REF-F | licence |
|---|---|---|---|---|
| **V-JEPA 2** (arXiv 2506.09985) and **V-JEPA 2.1** (arXiv **2603.14482**, 2026-03; repo `facebookresearch/vjepa2` @204698b) `PRIMARY` | V-JEPA 2: L **300M**@256, H **600M**@256, g **1B**@256 / @384. 2.1: **B 80M**, L 300M, g 1B, G 2B (all @384; B and L distilled from G) | **yes** (video encoder; >1 M h video; 2-AC post-trained on < 62 h Droid, zero-shot Franka planning with image goals — SECONDARY) | 2.1 adds a *dense predictive loss* → "high-quality and temporally consistent dense features" (README). The only candidate that sees **time**, which the frozen single-frame DINOv2/I-JEPA failures (`INHERITED (brief)`) say is missing | MIT (parts Apache-2.0) |
| **DINOv3** (arXiv 2508.10104; repo @6876159) `PRIMARY` | distilled ViT-S 21M · S+ 29M · B 86M · L 300M · H+ 840M; ViT-7B 6,716M (teacher); ConvNeXt T/S/B/L (distillation code released 2025-11-20) | no (image) | dense features via Gram anchoring; frozen 7B: 55.9 mIoU ADE20k linear (SECONDARY). The DINOv2 metric-ego-motion failure (brief) is the prior to beat | *DINOv3 License* (Meta custom; royalty-free limited licence; publications must acknowledge) — **commercial terms UNVERIFIED, read `LICENSE.md` fully before deployment** |
| **SigLIP 2** (arXiv 2502.14786) `SECONDARY` | family incl. **NaFlex** (variable resolution, native aspect ratio); Qwen3-VL uses **SO400M (400M)** and **Large (300M)** towers; full size table not read | no (image–text) | **the frozen backbone CoVer-VLA actually uses** (`PRIMARY`: open_clip SigLIP 2, bf16, ViT-B/16 patch grid per code comment); language-aligned, not metric | Apache-2.0 *recalled, UNVERIFIED* |
| **Qwen3-VL** (arXiv 2511.21631) `SECONDARY` | dense **2B / 4B / 8B / 32B**; MoE 30B-A3B / 235B-A22B; 256K interleaved text–image–video context; SigLIP-2 vision tower | video-capable VLM | same LLM family as CLM's encoder (Qwen3-8B, `PRIMARY`); too heavy as a per-tick encoder, plausible as a System-2 component | Apache-2.0 *recalled, UNVERIFIED* |
| **NVIDIA Cosmos** — Embed1, Reason1/2 `SECONDARY` | **Cosmos-Embed1** = dual-encoder video–text embedder: EVA-ViT-G @224 on sampled frames → Q-Former (BERT-init) → one clip vector; BERT text tower. Reason = VLMs (Reason2: **2B** per the Jetson tutorial title; other sizes not verified) | Embed1 consumes sampled video frames | Embed1 is built for retrieval / dedup / k-NN — a **single pooled vector**, i.e. CLM's own pooled-vector regime that CoVer avoids; Jetson AI Lab has a **Cosmos-Reason2 (2B) tutorial for Thor/Orin** — numbers unread | NVIDIA licences, UNVERIFIED |
| **C-RADIOv4** (arXiv **2601.17237**, released 2026-01-27) / C-RADIOv3 / RADIOv2.5 (2412.07679) / AM-RADIO (2312.06709); repo `NVlabs/RADIO` @c0f3701 `PRIMARY` | v4: **SO400M**, **ViT-H/16**; v3: B, L, H, g | no; `cpe_video_mode(t)` encodes B·T frames with a shared positional viewport — frame-wise, **no temporal attention** | multi-teacher student: v4 teachers **SigLIP2-g-384, DINOv3-7B, SAM3** (one forward ≈ three backbones); E-RADIO "6–10× faster than CLIP and DINOv2" (README); `model_results.csv` throughput RADIOv2 H/16 **556 img/s**, E-RADIO **3,697 img/s** (GPU unspecified) | **C-RADIO models: NVIDIA Open Model License (commercially permissive)**; RADIO/E-RADIO: NSCL non-commercial |

### 4.1 Compute and cache arithmetic (`ESTIMATED`; linear-layer FLOPs ≈ 2·params·tokens; attention adds ≈ 12 / 33 / 54 GFLOP for B / L / H at 576 tokens)

| encoder | tokens | GFLOP per image | note |
|---|---|---|---|
| ViT-S 21M / B 86M / L 300M | 576 | **24 / 99 / 346** | 384×384 at patch 16 |
| SO400M (SigLIP-2 style) | 729 (384, patch 14) | **583** | |
| ViT-H ≈ 632M / g ≈ 1B / 7B | 576 | **728 / 1,152 / 7,737** | 7B is a teacher, not an edge model |
| V-JEPA-style clip, 16 frames @256, tubelet 2 | 2,048 | L **1,229** · g **4,096** per clip | video-native cost is per *clip* |

Cost per tick = cameras × frames × the row (camera count for REF-F is unspecified in the brief — **not assumed**). **Token cache** for "train heads on cached
embeddings" (CLM's economy): 576 × 768 × fp16 = **0.88 MB/frame** (1.18 MB at 1,024-d, 1.47 MB at 1,280-d) — the economy survives only if the cached object is
small; CLM caches *one 4,096-d vector*, which is exactly what fails for metric ego-motion.

### 4.2 Published edge latency actually found (thin — treat as *absence* of usable numbers)

* `PUBLISHED-SECONDARY` — embedl-deploy docs ("FP8 ViT on NVIDIA Jetson AGX Thor", TensorRT 10.13.3): median **1.065 ms (FP8) vs 1.399–1.402 ms (FP16)**, 1.32×.
  **ViT variant, resolution and batch not visible to me (page egress-blocked) — do not extrapolate.**
* `PUBLISHED-SECONDARY` — Orin **Nano**: ViT-B/16@224 **17.0 ± 4.2 ms (FP32) / 19.9 ± 3.6 ms (FP16)**; ViT-B@384 **49.2 ± 7.5 / 40.7 ± 5.9 ms**; the search summary attributed
  these to a benchmarking paper (arXiv 2607.11356 was the top hit) — **attribution UNVERIFIED**.
* `PUBLISHED-SECONDARY` (community, second location probed) — GitHub search surfaced `DylanPina/foresight-orin-thor-bench @1ab1bd4` (vLLM vs TensorRT Edge-LLM VLM benchmark on Orin and Thor; workload = 40 COCO images @336×224, 512 max tokens, Qwen3.5-0.8B/2B and a Qwen3-VL-2B SFT) and `VitalyAnkh/thor-sm110-microbench @1fc0fc8` (instruction-level Thor SM_110 microbenchmarks). **I cloned both: neither commits any end-to-end result table** (the former's `data/` is not in the repo; the latter reports instruction latencies/throughputs, not model latency) — pointers, not numbers.
* `NOT READ` — `jetson-ai-lab.com` Thor/Orin benchmark tables and the TensorRT Edge-LLM performance page (egress-blocked); they may hold Qwen3-VL / Cosmos-Reason2 rows.
* `INHERITED (CLAUDE.md, MEASURED 2026-08-03 on TanitAD's own workload)`: on Thor only in-process `torch.cuda.max_memory_allocated()` is an admissible memory probe;
  throughput was flat at 12.3–14.1 windows/s over a 6× batch range (20 SMs saturate near batch 8); each dataloader worker costs ~8.6 GB host RAM.
  ⇒ **latency for every candidate above must be MEASURED on Thor with the same harness before it is quoted.**

### 4.3 Suggested order to test (`HYPOTHESIS`, gated by E-F0)

1. **V-JEPA 2.1 ViT-B then ViT-L** — video-native, 80M/300M; addresses the "single-frame" failure class directly.
2. **C-RADIOv4-SO400M / -H** — strongest *commercially licensed* frozen student (SigLIP2 + DINOv3 + SAM3 knowledge); image-only, so pair with stacked frames / ego channels.
3. **SigLIP 2 B/L** — CoVer's own backbone; the like-for-like ablation against the published verifier.
4. **DINOv3 ViT-B or ConvNeXt** — dense-feature control against the DINOv2 failure.
Always with **trainable attention pooling over patch tokens**; never a single frozen pooled vector. Qwen3-VL / Cosmos-Reason are System-2 candidates, not per-tick encoders.

**Named picks for the plan's H-F7 slots** (`REFF_CONTRASTIVE_SELECTOR_PLAN.md` §2.2 defers *"specific checkpoints"* to this stream's latency-and-licence check; the plan's default backbone is TanitAD's own frozen v1 encoder, `F-own`):
**F-vid = V-JEPA 2.1 ViT-B/16 (80M @384, MIT)**, ViT-L/16 (300M) as the scale-up; **F-img = C-RADIOv4-SO400M (NVIDIA Open Model License, multi-teacher)**, with **DINOv3 ViT-B/L** as the dense-feature control (custom licence — read it) and
**SigLIP 2 B/L** as the CoVer-comparable arm. **Latency status: none published for any of them on Thor** — shortlist by 4.1 (B ≈ 99, L ≈ 346, SO400M ≈ 583 GFLOP/img at 576–729 tokens) and **measure**; licence status per the table above (two UNVERIFIED).

---

## 5. Supporting material — access log, toy code, citation status

### 5.1 Access log and budget (fail-loud)

* **WebSearch: 25 of 25 used.** **WebFetch: 6 attempted, 1 succeeded** (`github.com/Oaklight/krino/issues/76`); **5 EGRESS_BLOCKED:** `contrastive-lm.notion.site`, `www.alphaxiv.org`, `implicitbc.github.io`,
  `nvidia.github.io`, `docs.embedl.com`. arXiv, Hugging Face, VentureBeat are blocked by policy, so **no arXiv PDF was read this session**.
* **`git clone --depth 1` of eight public GitHub repos** — CLM, cover-vla, vjepa2, dinov3, RADIO, google-research/ibc and the two community Thor benchmark repos of 4.2 (github.com is egress-permitted; done so
  code is read verbatim rather than through a summariser; this is *not* a WebFetch and is disclosed here rather than hidden). One further attempt, `google-research/contrastive_rl`, failed (repository not found under that
  name), so Contrastive RL stays `SECONDARY`. GitHub MCP `search_repositories` / `search_code` (read-only, global) found the community repos and returned nothing for the two narrower Thor-latency queries.
  The GitHub MCP `get_file_contents` refused the CLM repo (session scoped to `sayoood/tanitad`); I did **not** add it.
* Everything tagged `SECONDARY` is a search-engine summary (one hop). Where two summaries disagreed with a primary source, the primary won (krino#76's DeepSWE/TB numbers, 1.4).

### 5.2 Toy code (verbatim; `MEASURED (ours, toy)`; numpy only; each block runs standalone, executed as written before staging)

```python
# B1: bimodal decision scenes; exact Bayes-optimal InfoNCE critic f* = log p(a|s)/p(a) (+c(s))
import numpy as np
rng = np.random.default_rng(0); K, S = 64, 4000
cr, rare = np.arange(8), np.arange(8, K); steady = rng.random(S) < 0.74
P = np.zeros((S, K))
for s in range(S):
    c, r = rng.choice(cr), rng.choice(rare)
    if steady[s]: P[s, c] = 0.97; P[s, rng.choice(rare, 3, replace=False)] += 0.01
    else:         P[s, c] = 0.60; P[s, r] = 0.40
pa = P.mean(0); pmi = np.log((P + 1e-12) / pa)
mode = P.argmax(1); raw = pmi.argmax(1); fix = (pmi + np.log(pa)).argmax(1)
err = lambda pick: np.mean([(P[s] * np.abs(pick[s] - np.arange(K))).sum() for s in range(S)])
print(pa[cr].sum(), np.mean(raw[steady]==mode[steady]), np.mean(raw[~steady]==mode[~steady]), err(raw), err(fix), err(mode))
# -> 0.874  1.000  0.000  5.58  3.95  3.95
# B2: negative control (shift-family Laplace conditionals): argmax f* == MAP in 4000/4000 scenes, lambda in {0.5,0.8,1.5,3.0}
x = np.arange(K, dtype=float); prior_c = np.full(K, 0.26 / 56); prior_c[28:36] = 0.74 / 8
centers = rng.choice(K, size=S, p=prior_c)
for lam in (0.5, 0.8, 1.5, 3.0):
    Q = np.exp(-np.abs(x[None] - centers[:, None]) / lam) + 1e-4; Q /= Q.sum(1, keepdims=True)
    print(lam, int((np.log(Q / Q.mean(0)).argmax(1) == Q.argmax(1)).sum()), "/", S)
```

```python
# B3: exact check that InfoNCE's optimum is log p(a|s) - log q(a) (+c(s)); bidirectional: log p(s,a) - log qa - log qs (+k)
import numpy as np, itertools
rng = np.random.default_rng(1); S, A, K = 3, 4, 3
J = rng.dirichlet(np.ones(S*A)*0.7).reshape(S, A); ps, pa = J.sum(1), J.sum(0); pas, psa = J/ps[:, None], J/pa[None, :]
qa, qs = rng.dirichlet(np.ones(A)*1.5), rng.dirichlet(np.ones(S)*1.5)          # arbitrary negative proposals
def pack(anchor_p, pos_p, q, n_anchor, n_cand, cell):    # enumerate anchor, positive and K-1 negatives exactly
    rows = [(anchor_p[x]*pos_p[x][y]*np.prod(q[list(n)]), [cell(x, y)] + [cell(x, k) for k in n])
            for x in range(n_anchor) for y in range(n_cand) for n in itertools.product(range(n_cand), repeat=K-1)]
    return (np.array([r[0] for r in rows]), np.array([[c[0] for c in r[1]] for r in rows]), np.array([[c[1] for c in r[1]] for r in rows]))
P1 = pack(ps, pas, qa, S, A, lambda s, a: (s, a))        # s->a: anchor s, candidate actions ~ qa
P2 = pack(pa, psa.T, qs, A, S, lambda a, s: (s, a))      # a->s: anchor a, candidate states ~ qs
def fit(packs, iters=60000, lr=2.0):
    F, v = np.zeros((S, A)), np.zeros((S, A))
    for _ in range(iters):
        G = np.zeros((S, A))
        for w, si, ai in packs:
            z = F[si, ai]; e = np.exp(z - z.max(1, keepdims=True)); g = e/e.sum(1, keepdims=True); g[:, 0] -= 1
            np.add.at(G, (si.ravel(), ai.ravel()), (w[:, None]*g).ravel())
        v = 0.9*v - lr*G; F = F + v
    return F
F1, F2 = fit([P1]), fit([P1, P2])
T1 = np.log(pas) - np.log(qa); T2 = np.log(J) - np.log(qa) - np.log(qs)[:, None]
print(round(np.abs((F1-F1.mean(1,keepdims=True))-(T1-T1.mean(1,keepdims=True))).max(), 4), round(np.abs((F2-F2.mean())-(T2-T2.mean())).max(), 4),
      bool((F2.argmax(1)==T1.argmax(1)).all()), int((T1.argmax(1)!=pas.argmax(1)).sum()))
# -> 0.0 0.0 True 2   (max abs deviations to 4 d.p.; ~6 s)
```

### 5.3 Citation verification status (this session)

`CONFIRMED` = arXiv id and title both appeared in a search-result list (or a repo's own BibTeX). `V1` = confirmed by the 2026-09-25 V1 citation check. `INHERITED` = from a prior repo doc only.

| item | id | status |
|---|---|---|
| CLM (Kwok et al. 2026) | Notion blog, **no arXiv id** (README BibTeX `@misc`, note "Notion Blog") | code/README `PRIMARY`; blog **NOT READ** |
| CoVer-VLA | arXiv 2602.12281 | CONFIRMED (result list + repo BibTeX/README) |
| VLA-R | arXiv 2511.12405 | CONFIRMED (result list; content SECONDARY) |
| IBC / Diffusion Policy | 2109.00137 / 2303.04137 | CONFIRMED (result URLs); IBC loss/negatives `PRIMARY` from code `@db89ddb`; `cd_kl` cites arXiv 2012.01316 (id from code comment only) |
| Contrastive RL / 1000-layer RL / CDPC | 2206.07568 / 2503.14858 / 2310.20141 | CONFIRMED / CONFIRMED / id from URL only |
| VINN | 2112.01511 | CONFIRMED; **Behavior Retrieval: NOT FOUND — unverified, unused** |
| DriveVLM(-Dual) / FASIONAD / ETA | 2402.12289 / 2411.18013 / 2506.07725 | CONFIRMED / CONFIRMED / INHERITED only |
| VADv2 · Hydra-MDP · Hydra-MDP++ | 2402.13243 · 2406.06978 · 2503.12820 | INHERITED (PDF-VERBATIM / HTML-SUMM); Hydra-MDP V1 ✓ |
| GTRS · DriveSuprim · WoTE · CLOVER | 2506.06664 · 2506.06659 · 2504.01941 · 2605.15120 | V1 ✓ (first three); GTRS + CLOVER also in this session's result lists |
| PDM-Closed · LLM-Assist | 2306.07962 · 2401.00125 | INHERITED (PDF-VERBATIM) |
| CPC · Poole et al. | 1807.03748 · 1905.06922 | CONFIRMED (id + title); *statements RECALLED* |
| Ma & Collins (EMNLP 2018) | ACL D18-1405 | CONFIRMED; **arXiv id UNVERIFIED** (a summary echoed my own query) |
| Menon (logit adjustment) · Yi et al. (logQ) · "Correcting the LogQ Correction" | 2007.07314 · RecSys'19 doi 10.1145/3298689.3346996 · 2507.09331 | CONFIRMED · CONFIRMED · title/id only |
| Weller et al. · ColBERT | 2508.21038 · 2004.12832 | CONFIRMED · CONFIRMED |
| Chuang (debiased CL) · Robinson (hard negatives) · soft-target NCE | 2007.00224 · 2010.04592 · 2404.14076 | CONFIRMED · CONFIRMED · id from URL, unread |
| Guo (calibration) · Angelopoulos & Bates (conformal) | 1706.04599 · 2107.07511 | CONFIRMED · CONFIRMED |
| V-JEPA 2 · V-JEPA 2.1 | 2506.09985 · 2603.14482 | CONFIRMED · repo BibTeX `PRIMARY` |
| DINOv3 · SigLIP 2 · Qwen3-VL · Qwen3 | 2508.10104 · 2502.14786 · 2511.21631 · 2505.09388 | CONFIRMED (all four) |
| C-RADIOv4 · RADIOv2.5 · AM-RADIO | 2601.17237 · 2412.07679 · 2312.06709 | repo README `PRIMARY` |
| Cosmos-Embed1 / Cosmos-Reason | NVIDIA pages (NGC, research.nvidia.com, Jetson AI Lab) | **no arXiv id seen**; SECONDARY |
| Jev / TypeSafe; CLM-35B-A3B | vendor blogs (MindStudio, DataCamp, LangChain, Analytics India); Sanity / pasqualepillitteri.it | SECONDARY, vendor claims |

### 5.4 Where this file corrects or qualifies the orchestrator's plan draft (`REFF_CONTRASTIVE_SELECTOR_PLAN.md` @4ad6529) — read-only comparison, nothing edited

| plan statement | what the primary sources show | class |
|---|---|---|
| §1 "same-task pairs masked" | `_clm_loss` masks only equal **`(task_id, step_idx)`** — other rollouts of the *same state step*; all other same-task pairs stay negatives (1.3-1) | PRIMARY (code) |
| §1 / §3.2 "Choice mode = softmax CE with soft targets" | CLM's `choice` default is **`--loss infonce`** (in-batch over distinct option texts); **`softce`** is the alternative. The plan's `L_voc` is CLM's `softce`, not its default; which trained the released head is UNVERIFIED (1.3-2) | PRIMARY (code) |
| §1 "about 20 M parameters" (heads) | default cfg = **9.44M per head / 18.89M for the pair = 75.55 MB fp32**, matching the README's "75 MB"; "2 × 20M" is not established (1.3-4) | MEASURED (ours, arithmetic) |
| §1 "cold request 58.1 ms … p50 28.6–28.8 ms for 3–50 new actions" | 58.1 ms is one README example (cold cache, 106 tokens). The table's 28.0–28.8 ms is **a new *state* every call** (action set fixed); 1.7 → 0.6 ms is a *revisited* state (1.3-3) | PRIMARY (README) |
| §1 "a secondary summary attributes 81.6 % to the baseline" | the **primary figure agrees with the README** (CLM 81.6 / 87.6, Jev 71.1 / 83.1, pass@1 73.7 / 84.0); krino#76 mislabelled the dashed pass@1 line as CLM (1.4) | PRIMARY (figure) |
| the plan and pre-registration cite no robotics precedent (no CoVer-VLA, IBC, Diffusion Policy, Contrastive RL or VINN) | **CoVer-VLA (2602.12281)**, same first author, is the closest published design and already matches the plan's own choices (frozen vision backbone, attention-pooled patch tokens, numeric action tower, symmetric InfoNCE); the plan's additions — residual vocabulary, prior correction, cost heads — are *not* in CoVer, so CoVer is the natural published comparison arm, and IBC / VINN / Contrastive RL supply precedents and baselines the plan lacks (2.1–2.3, 2.5) | PRIMARY (code) |
| §2.5 "InfoNCE over-selects rare manoeuvres" (H-F4) | true in the exact-critic algebra and in the plan's toy; **but** a shift-family control shows zero flips in 4,000/4,000 scenes — the bias needs multimodal conditionals with globally-rare secondary modes, so H-F4's *refuted* branch is live, not a formality (3.1) | MEASURED (ours, toy) |
| §4.2 H-F3 "recovers ≥ 25 % of the oracle gap" | registry §4.1: learned re-scorers **≤ 8.4 %** (47 arms, own training data), v1.2 **+2.9 % n.s.**; the only ≥ 25 % evidence is E-GOAL-4's 32–36 %, **out-of-fold within val-600 only** (programme review) — the bar should be stated as "paired against Δ2" (§6 #3) | MEASURED (registry) + INHERITED |
| §3.3 stage-2 near-miss negatives | consistent with CLM (hard negatives *after* pre-training: 52.1 → 69.2 % vs 62.4 % from the start). IBC's own code adds a **near-duplicate mask** for close counter-examples (`clipped_cd`) — precedent for extending the plan's ε-mask from `L_nce` to stage 2 (3.4) | PRIMARY (README, code) |

---

## 6. Implications for REF-F (≤ 15)

1. **Copy CoVer, not CLM's backbone** (the plan draft's §2.2–2.3 already does; CoVer is its missing citation and comparison arm, 5.4). The closest published system is CoVer-VLA (2602.12281, same authors): frozen SigLIP 2 + *trainable pooling over patch tokens* + a **numeric**
   action-chunk tower + symmetric InfoNCE + verifier ensemble. CLM-8B's frozen Qwen3-8B *text* encoder with last-token pooling is the wrong substrate for numeric trajectories
   and metric ego-motion. `PRIMARY (code)`; recommendation `HYPOTHESIS`.
2. **Scope REF-F as a verifier over REF-C's fan, not a planner.** CLM cannot generate — it ranks *given* candidates (zero-shot: small sets, 3 actions in T-Rex; SOTA claims: best-of-N verification; CoVer: best-of-40
   verification). CLM's only control-like demo (T-Rex) is label-reading with **65.8 %** agreement and **≤ 29.2 %** shield interventions — zero closed-loop evidence. REF-F's "driving command" is therefore an argmax over a
   candidate set plus a controller: **its ceiling is that set's oracle** (REF-C-XL 256-anchor fan: 0.1640, `MODEL_REGISTRY.md` §4.1). `PRIMARY + MEASURED (ours)`.
3. **Pre-register against the registry's ceiling, not against ADE headroom.** `MODEL_REGISTRY.md` §4.1: selected **0.4714** vs oracle-in-fan **0.1640**; learned re-scorers recovered
   **≤ 8.4 %** of the oracle gap on their own training data across 47 arms; v1.2 **+2.9 %, not significant**. The E-GOAL-4/Δ2 expected-cost selector (0.5015 → 0.3917, out-of-fold
   *within* val-600, `INHERITED (programme review 2026-09-25)`) is **unreconciled** with that. REF-F's claim should be "matches Δ2 at lower latency / on unseen candidates", paired against Δ2.
4. **A one-hot InfoNCE positive is a hard-argmin target — the registry's worst in all five configurations; VADv2 loses 17× without its soft loss.** Stage 2 = softmax-CE over the
   actual fan with **soft distance-weighted** targets; InfoNCE only for stage-1 representation. `MEASURED (registry) + INHERITED`.
5. **The InfoNCE argmax is PMI, not likelihood** (3.1). Under a ~74 % cruise prior pair it with `+log q̂(a)`, uniform-over-the-fan negatives, or a listwise stage 2 — unless E-F1 shows the bias is immaterial. Toy: 0 % MAP agreement on
   ambiguous scenes (5.58 vs 3.95 error), but **4,000/4,000 agreement** for shift-family conditionals — so run **E-F1**; CLM's code has no correction. `MEASURED (toy) + PUBLISHED-RECALLED`.
6. **Scores are comparable only within a state; CLM's softmax is not a calibrated confidence.** Calibrate *after* the prior fix (temperature scaling → episode-cluster-bootstrap ECE);
   gate System-2 hand-off with an **episode-split conformal set**. `PUBLISHED + HYPOTHESIS`.
7. **Negatives: copy CLM's curriculum, with soft positives.** In-batch → fan hard negatives → in-domain replay: hard negatives from step 0 peak **62.4 %** vs **69.2 %**; agentic-only drops
   to **56.2 %**, 40 % replay keeps **68.5 %**. First **measure** the in-batch false-negative rate (pairs with GT ADE ≤ δ) before fixing the batch size. `PRIMARY (README) + HYPOTHESIS`.
8. **Dual encoder = stage 1 until E-F4 says otherwise.** Rank-≤512 bilinear scores vs interaction-heavy costs; retrieve-then-rerank (DriveSuprim **+3.6 PDMS**; registry K = 8–32 plateau);
   **per-family sub-scores** (Hydra: scalar 80.2 vs five heads 83.0) — which the four-family eval rule mandates anyway. `PUBLISHED + INHERITED`.
9. **CLM's latency headline does not transfer.** 0.6 ms needs a *repeated* state; a driving tick is a new state every 100 ms — the measured cold path is **≈ 28 ms p50** on an RTX 4090 for an
   8B *text* encoder on short text. What is cheap is O(K) candidate scoring against a cached vocabulary and **generalisation to unseen candidates** (GTRS +11.1 EPDMS), so the state tower must be
   small. Thor latency is unpublished for every candidate ⇒ **measure** (only `torch.cuda.max_memory_allocated()` is admissible). `PRIMARY + estimate`.
10. **Backbone order (`HYPOTHESIS`):** V-JEPA 2.1 B/L → C-RADIOv4 → SigLIP 2 → DINOv3, always with trainable patch-token pooling. **E-F0** (k-NN over the frozen embedding vs hold-speed on
    non-steady windows) gates all of it: frozen single-frame DINOv2/I-JEPA already failed as metric ego-motion substrates (`INHERITED (brief)`).
11. **Evaluate on the four families, steady vs non-steady split, and adopt CLM's selected / random / oracle triple plus a "decidable windows" count** (`bon_eval.py` computes `mixed_tasks`;
    the README omits it). CLM's verifier gains (**+7.9 / +3.6 pts** over pass@1) rest on **38 / 30** tasks (Wilson-95 % for 31/38 = **[0.666, 0.908]**) — direction, not effect size. `MEASURED (ours)`.
12. **Ego state:** BEV-Planner / AD-MLP-type evidence in the repo's scorer-inputs survey (`INHERITED`) says ego status alone matches learned stacks — a scorer given v0 will learn "hold speed".
    Keep ego state in the *state tower only*, benchmark against hold-speed, and **confirm with the PI** that ego state at inference is admissible for REF-F given the 2026-08-03 vision-only ruling
    (written for the scenario classifier; its generalisation rule says to ask for ANY head). `ESCALATION`.
13. **Scaling-law transfer is inadmissible:** N\*∝D^1.02 has no window/R²/n and was fit on text QA; report REF-F's own matched-step ratios. **CLM-35B-A3B** (multimodal MoE, "early October",
    unreleased as of 2026-09-29) is a backbone *to test later*, not a plan dependency. `PRIMARY + SECONDARY`.
14. **Two unread sources block the strongest claims:** the CLM blog (scaling exponents, `clm-raw` and Choice-vs-InfoNCE ablations, failure cases) and the HF model card/config (head size 2×9.4M vs
    2×20M). **Someone with unblocked access should read both before any GPU-day rests on a CLM-recipe claim.** `UNVERIFIED gap — ESCALATION`.
15. **Sequence:** E-F0, E-F1, E-F2 ride on the same parity-train fan dumps Δ2 already needs (programme review D2a); do not enter REF-F in the registry before Δ2 reports. `ESTIMATED`.

---

## 7. Deliverable manifest

| artifact | where it lives | note |
|---|---|---|
| this stream file `2026-09-29-reff-prior-art-and-theory.md` | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-reff-prior-art-and-theory.md` — **staged** (`git add`; verified with `git ls-files --stage`); **not committed** | **exists in ONE place** (this working tree + index) until the orchestrator commits |
| toy scripts `pmi_toy3.py`, `pmi_control.py`, `b3.py`, `infonce_check.py` (B1/B2/B3 embedded verbatim in §5.2) | scratchpad only (`…/scratchpad/toy/`), disposable | reproducible from §5.2; `pmi_toy.py`, `pmi_toy2.py` were exploratory (their finding — shift-family conditionals show no bias — is folded into the control) |
| read-only clones of CLM `@bb42c6c`, cover-vla `@21a9960`, vjepa2 `@204698b`, dinov3 `@6876159`, RADIO `@c0f3701`, ibc `@db89ddb`, foresight-orin-thor-bench `@1ab1bd4`, thor-sm110-microbench `@1fc0fc8` | scratchpad only; **not staged** (third-party code) | re-clonable from github.com |

**Integration escalation:** none to merge; **decisions needed** — (i) PI: approve/reject E-F0…E-F4 pre-registrations (3.6); (ii) PI: ego-state admissibility for REF-F (§6 #12); (iii) whoever has unblocked
egress: read the CLM blog and HF card (§6 #14).
