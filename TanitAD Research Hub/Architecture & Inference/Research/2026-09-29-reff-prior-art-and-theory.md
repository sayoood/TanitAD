# REF-F prior art, theory and backbones — Stream F-R (2026-09-29)

**Status:** IN PROGRESS — banked incrementally (section 1 first). Scope: transfer of **CLM-8B**
(Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré, Mirhoseini; Stanford + NVIDIA; released 2026-09-23;
Apache-2.0) to driving as reference arm **REF-F** (camera + ego state in, driving command out).
**No TanitAD model number is quoted from this file's sources** — every TanitAD figure below is tagged
`INHERITED (orchestrator brief; registry not re-read)`; the registry wins on any conflict.

**Evidence classes:** `PUBLISHED-PRIMARY` (paper/README/code/figure I read myself) ·
`PUBLISHED-SECONDARY` (press/vendor/third-party summary — one hop, re-verify before spending a GPU-day) ·
`INHERITED` (another repo doc, not re-verified) · `MEASURED (ours)` (computed here; script named) ·
`ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

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
| serving | encoder = vLLM Qwen3-8B pooling server, `--max-model-len 2048` (states truncated, **tail kept**; actions **head kept**); heads 512-d; LRU "vector arena" 2 % of device memory keyed by head generation |
| `clm-raw` | shipped **ablation**: cosine in the raw encoder space, scale fixed 100, no head (numbers are in the blog only) |
| trajectory verifier (`bon_eval.py`) | per-step cosine score → **mean over final `--window 12` steps** → pick argmax per task; reports **selected / random / oracle** with exact expectation over uniform N-subsets and uniform tie-breaking |

### 1.3 Corrections / refinements to the brief (each checked against source)

1. **False-negative mask is per *state step*, not per task.** `codes` groups rows by `(task_id, step_idx)`;
   only *other rollouts at the same step of the same task* are masked. Everything else from the same task
   stays a negative (deliberately: `task-blocked` batches exploit this). *(finetune.py `_load_clm_data`, `_clm_loss`.)*
2. **Choice mode default is InfoNCE, not softmax-CE.** `softce` exists as the alternative; which one trained
   the released `clm-latest` head is **UNVERIFIED** (the released head comes from the 3-stage pipeline, not `--task choice`).
3. **"cold 58.1 ms" is one README example, not the latency table.** The measured table (RTX 4090, server p50,
   fixed action set) is **28.0–28.8 ms for a new state every call** and **0.6–0.7 ms for a revisited state**;
   the T-Rex run shows `model_ms_p50` **2.6 ms** and client-side p50 **16.5 ms**. The 0.6 ms figure needs a
   *repeated* state. **A driving loop never revisits a state (a new frame every 100 ms)** ⇒ the state side pays the
   full encoder every tick; only the action side (fixed vocabulary/anchors) caches. (ESTIMATED consequence.)
4. **Trainable-parameter count is ambiguous.** README: "20M-parameter head" per encoder. Default cfg
   (width 1536, depth 3, LN) = **9,443,840 params/head, 18,887,680 for the pair = 75.55 MB fp32** —
   *exactly* the README's "75 MB reference head". Either the "20M" is the **pair**, or the released cfg is wider and
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
  `mixed_tasks` but the README does not print it). This is exactly TanitAD's oracle-gap framing — **CLM publishes
  the selector-vs-random gap but not the selector-vs-oracle gap.**

**T-Rex is not evidence of control skill (read the harness before quoting it).** `examples/t_rex/README.md`:
a hand-written physics planner labels each action *safe/unsafe/best* **inside the candidate texts** ("jump: Safe.
Clears the 2 large cacti. Best."), so the model reads the answer; a **shield** replaces unsafe answers and an
emergency check can act first. From the per-seed rows: CLM **agreement with the planner's best move 65.8 %**
(Jev **98.7 %**); interventions (shield + arrival + emergency) **4,883 / 16,709 decisions = 29.2 %** (Jev
**28 / 5,595 = 0.5 %**). Both survive because of the shield. `MEASURED (ours)` from the published JSON
(`summary.shield_interventions` = Σ shield 364 + Σ arrival-saves 4,519). ⇒ the *only* closed-loop use in the repo is a
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
* **CLM-35B-A3B** (multimodal MoE): "early October 2026" per two secondary sources; the README roadmap says only
  *"Vision and multimodal support: images, video and other modalities for robotics and computer-use tasks."*
  **As of 2026-09-29 it is announced, not released** — `PUBLISHED-SECONDARY`, UNVERIFIED against the blog.
* **Robotics / driving:** none published. Marco Pavone (Stanford/NVIDIA AV research) is a co-author and the roadmap names
  robotics, but two targeted searches and the repo contain **no driving or robot result**. REF-F would be the first
  public transfer — i.e. there is **no external number to calibrate expectations against**.

<!-- SECTIONS 2-5 + MANIFEST: pending (banked next) -->
