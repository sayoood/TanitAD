# HiCAP stream R-H — CoVer-VLA from primary sources, and what HiCAP should take from it (2026-10-26)

**Status:** COMPLETE. Staged, **not committed**. This stream wrote only this file. No HiCAP or REF-F file was edited; the paper corrections it found are listed under *Escalations*.

**Sources read.** Every one is primary, and each is pinned to a commit or a hash.

| source | pin | how it was read |
|---|---|---|
| code `github.com/cover-vla/cover-vla` | HEAD **`21a9960639a8340c1e8511fb04776fb11271f246`** | shallow clone, read only, nothing executed |
| code `github.com/cover-vla/cover-vla-polaris` (the PolaRiS / DROID companion, linked from the main README) | HEAD **`2cba5bc85841c9d9c6d9cb96c95a2d68cd60ff79`** | shallow clone, read only |
| paper arXiv 2602.12281 **v2**, full text | `https://huggingface.co/papers/2602.12281.md` (an HF mirror of `arxiv.org/html/2602.12281v2`; arXiv itself is proxy-blocked) | markdown. **Tables 1, 2, 4, 5, 6, 7 and Figures 2, 5, 6, 8 are images and are not in the text.** Any number that exists only in those images is marked *not in text* and is not quoted |
| weights `cover-vla/cover-vla-bridge/cover_verifier_bridge.pt` | 316,400,837 B, sha256 `8842ec9d…bdf8b31` (equal to the HF LFS oid) | state-dict **keys and shapes only**, read with a restricted unpickler that never imports torch or repo code (Appendix A) |
| weights `cover-vla/cover-vla-polaris/cover_verifier_polaris.pt` | 319,748,154 B, sha256 `0738d26c…1bc3380` (equal to the LFS oid) | same method |

**Evidence classes** (CLAUDE.md rule 1):
- `PUBLISHED-CODE`: read in the code at the pinned commit, cited as file:line.
- `PUBLISHED-PAPER`: the paper text, cited by section.
- `MEASURED`: computed here from the checkpoint key and shape listing (Appendix A), or arithmetic on published numbers.
- `INHERITED`: a TanitAD document, not re-verified; registry citations are line numbers in `Project Steering/MODEL_REGISTRY.md` at `e8b040c`.
- `ESTIMATED`, `HYPOTHESIS`: as defined in CLAUDE.md.

File-path shorthands: `M` = `bridge_verifier/ensemble_eval/model.py`, `T` = `bridge_verifier/ensemble_eval/finetune_trajectory_bridge_ddp.py`, `E` = `bridge_verifier/ensemble_eval/efficient_ensemble_merged.py`, `R` = `CoVer_VLA/inference/experiments/robot/simpler/run_simpler_eval_with_openpi.py`, `U` = `…/simpler/eval_utils.py`, `PS` = cover-vla-polaris `third_party/openpi/scripts/serve_policy_with_verifier_Gemini.py`, `PD` = cover-vla-polaris `third_party/droid_verifier/efficient_ensemble_merged_droid.py`.

---

## Headline answers

1. **Architecture.** CoVer is a frozen **SigLIP 2 ViT-L/16 at 384 px, with both its image and its text towers in use**. Its trainable heads total **25.76 M parameters per verifier**, and an **ensemble of 3** is deployed, so **77.3 M are trainable in all** (MEASURED from the checkpoint).
   - **State side:** a parameter-free, *text-conditioned* attention of the instruction's tokens over the 576 patch tokens, followed by two 4-layer single-query attention poolers (vision and text, each 512-d) and a linear map from 1024 to 512. The **instruction text is a state input.**
   - **Action side:** a linear map from 7 to 512, then a 4-layer, 8-head, 512-d transformer encoder, then masked mean pooling over a **10-step window of 6 executed past actions and the 4-step candidate chunk.** In the Bridge checkpoint this window has **no positional encoding**, so the tower is order-invariant. The PolaRiS checkpoint does have one.
   - **Loss:** symmetric in-batch InfoNCE with a learned temperature initialised at 0.07. There are **no hard negatives** and no prior correction.
   - **Selection:** argmax of the cosine.
2. **Hierarchical selection.**
   - *Candidates:* K = 8 prompts (7 boot-time GPT-4o rephrases plus the active instruction), and **M = 5 π0 samples per prompt, so 40 candidates.*
   - *Prompt level:* the prompt whose 5 candidates have the highest **mean** score wins.
   - *Action level:* the best chunk within that prompt is executed.
   - **The released code adds two things the paper's Algorithm 2 does not contain.** First, a **confidence gate**: below cosine 0.1 on Bridge (it scores only one candidate) and below 0.45 on PolaRiS (the M candidates of the current prompt), and only then does it expand to all K×M. Second, a **"sticky" instruction**: the code scores candidates against the *previously selected* rephrase, not against the user's instruction as the paper states.
3. **Frequency.** CoVer verifies **once per chunk boundary** and executes the chunk open-loop in between. There is no fast path, no asynchronous refresh and no multi-rate scheme beyond the split between boot time and run time.

   | setting | control rate | chunk | decision period |
   |---|---|---|---|
   | Bridge / SIMPLER | 5 Hz | 4 steps | 0.8 s simulated time (1.25 Hz) |
   | PolaRiS / DROID | 15 Hz | 16 steps (server default) | ≈1.07 s |

   Wall-clock figures from the paper:
   - About **453 ms (≈2.2 Hz) for 16 candidates**, and **756 ms (≈1.3 Hz) at batch 32**, on an RTX 5090.
   - The π0 batch forward pass dominates, at 344 to 748 ms. The action tower adds about **7–8 ms**.
   - That is **3.4–7.5× HiCAP's entire 100 ms tick, before moving to Thor** (ESTIMATED).
4. **The lever transfers only at the level of modes.** CoVer's large gains are closed-loop **success-rate** gains on *grounding* failures, such as grasping the wrong can. Its own regression-error gain is modest: **−11 % RMSE at N = 16 versus N = 1** (PUBLISHED-PAPER §8.5.3). That is the same order as REF-C's learned re-scorer, which recovered **at most 8.4 % of the oracle gap**, a gap that is **~92 % irreducible** (INHERITED, registry l.1355–1360; the 8.4 % is itself a prose note). CoVer's test-time "scaling law" (Fig. 2) is an **oracle min-over-k error**, which is precisely the lottery statistic that the registry forbids reading as recoverable headroom.
5. **Top three proposals** (all HYPOTHESIS, §4):
   - **P1:** a CoVer-style verifier over REF-C-XL's 256-fan, decided on the **TACTICAL** family (did it select the right manoeuvre?), not on recovery of the ADE gap.
   - **P2:** **NC10-AP**, CoVer's attention-pooling aggregator as a third NC10 arm. It splits HiCAP minus CLM-pure into the share due to the aggregator and the share due to the added inputs.
   - **P3:** at zero extra training cost, ensemble the 3 seeds we already train, comparing **CoVer's embedding mean against a score mean**. CoVer's embedding mean contains 6 cross-model terms out of 9, and those are meaningful only if the seeds' spaces are aligned.

## Escalations (decisions and integration — for the orchestrator, not edits)

- **E1: three HiCAP paper sentences need correcting.** I did not edit them, per the brief.
  - (a) `sec_related.tex:28` says CoVer uses "attention pooling over patch tokens". It actually uses *text-conditioned* attention over patch tokens, then learned-query attention pooling, with the **instruction text as a state input**.
  - (b) `sec_method.tex:23` says "recent frozen-backbone work **favours** a learned aggregator over bare last-token pooling \citep{kwok2026cover}". **CoVer contains no ablation of pooling against CLS or last-token readout** (I searched the full text). It *uses* an aggregator; it presents no evidence *for* one. The sentence should drop "favours" or cite something that ran the comparison.
  - (c) The `sec_diff.tex:31` dagger reads "a frozen image encoder, not a full vision-language model". It should read "a frozen SigLIP 2 dual image–text encoder, not a generative VLM". The text tower runs at inference on the instruction.
- **E2: P1 cannot use Alpamayo 2 candidates.** The corpus stores the `nvidia/Alpamayo2-Super` trajectory task as **shape plus ADE/FDE, not coordinates**, and only at one anchor per clip (`Paper/HiCAP/design/HICAP_V10_SPEC.md:28–29`, INHERITED). A verifier over Alpamayo candidates would need Alpamayo re-run. P1 therefore runs on REF-C-XL's fan. Re-running Alpamayo is a PI compute decision.
- **E3: the anti-echo rule should be written into PREREG_HICAP before any verifier arm trains.** CoVer's code scores candidates against an instruction that its own selector chose in the previous chunk. That is the nav-echo class (CLAUDE.md BINDING 2026-08-03) in a third costume. Every HiCAP verifier arm should be pre-registered with the rule *"never score against a goal or prompt that the scorer itself selected; report the sticky-off control"*.

---

## 1. Exact architecture, from code (priority 1)

### 1.1 State encoder: frozen SigLIP 2, image *and* text

| element | fact | evidence |
|---|---|---|
| backbone | `hf-hub:timm/ViT-L-16-SigLIP2-384` (open_clip). This is the default for weights-only checkpoints and the training CLI default | `E:40–45`; `T:1414` (CLI default). ⚠️ The function signature default in `T:555` is `ViT-B-16-SigLIP2-256`, so the CLI and the function disagree. The checkpoint settles it: `pos_emb` is **576×1024** = (384/16)² patches × width 1024, which is ViT-L/16-384 (MEASURED) |
| frozen | `requires_grad=False` on every backbone parameter, cast to bf16 | `T:186–191`; `E:61–66` |
| which layer, image | a forward hook on **`visual.trunk.blocks[-1].attn`**. This is the **output of the attention sub-layer of the last ViT block, *before* that block's residual add and MLP**: not the block output and not the pooled embedding. The CLS token is dropped if present, then the features are L2-normalised per token | `T:271–274`, `T:336–354` |
| which layer, text | a hook on `text.transformer`, then `ln_final`, then `text_projection` applied **per token**, then L2 norm. Padding tokens are kept: no mask is passed to the poolers | `T:275–277`, `T:320–330`; mask `None` at `E:217–219` |
| text-aware visual extraction ("ClearCLIP-style", Otter) | similarity = ⟨text tokenᵢ, patchₖ⟩ (cosine, since both are normalised). Then softmax over patches at a **learned temperature initialised at 0.07** and clamped to [0, 100]. Then a weighted sum of (patch + fixed sincos 2-D-flattened position embedding). The output is **one visual token per text-token position**. Its only trainable parameter is the temperature | `M:50–73` |
| attention pooling (×2: vision and text) | `AttentionPooling(input 1024 → output 512, heads 8, layers 4, readouts 1)`: one learned 512-d query, 4 cross-attention blocks (kdim = vdim = 1024, pre-LN on the query, residual, then LN and an MLP of width 512 with a residual), and a final LN | `M:7–38`, `M:76–112`; dims at `E:110–127`; keys and shapes in Appendix A |
| fusion | concatenate [text token, vision token] (1024) → `Linear(1024, 512)` → L2 norm. This is $\mathbf F_{\text{combined}}$ | `T:233–235`, `T:372–375` |
| language | **the instruction is a state input.** The verifier scores (image, instruction, action) | `T:357–375`; paper §4.1, $\mathcal V_\theta(o_t,h_t,l,a)$ |
| PolaRiS variant | two views (exterior and wrist) share one text-aware extraction and one vision pooler. The projection is `Linear(1536, 512)` over [text, ext, wrist] | `PD:7–8`, `PD:287–305`; checkpoint `input_projection.weight` 512×1536 (MEASURED) |

### 1.2 Action encoder: a numeric chunk tower that also carries action history

| element | fact | evidence |
|---|---|---|
| input | a window of shape (H, D). Bridge: **H = 10 = up to 6 executed past actions plus the 4-step candidate chunk, D = 7** (Δxyz, Δrpy, gripper). Early steps are **left**-padded with −5.0 | `U:171–220` (6 past, `num_past = min(len, 6)`); `E:379–390`; padding value `T:281`; checkpoint `single_step_action_encoder.weight` 512×7 (MEASURED) |
| tower | `Linear(7, 512)` → `TransformerEncoder(4 layers, d 512, 8 heads, FF 1024, dropout 0.1, post-LN)` → **masked mean pool** over non-padded steps → L2 norm | `T:242–253`, `T:380–414`; `E:135–149`, `E:228–245` |
| ⚠️ positional encoding | **none on Bridge.** The tower is permutation-invariant over the 10 steps: it cannot tell past from future, or a chunk from its time-reversal, except through padding | `T:386–393` adds nothing; the Bridge checkpoint has **no** positional key (MEASURED). The **PolaRiS** checkpoint has a learned `action_pos_embedding` (1, 31, 512), added at `PD:317` |
| PolaRiS window | **H = 31 = 15 past + 16 future**, as the code comment states ("2·16 − 1"). **D = 8** (joint-position deltas plus gripper, quantile-normalised) | `PS:104`; `PD:52`; checkpoint `single_step_action_encoder.weight` 512×8 (MEASURED) |
| alternative | an MLP over the flattened window, (H·D → 512 → LN → ReLU → 512). Not used by the released checkpoints | `T:255–262` |

The history is on the **action** side ($\mathbf A(h_t,a_t)$, paper Alg. 1 l.13). So **action embeddings depend on the episode and cannot be cached.** Each decision re-embeds every candidate. This is the main structural difference from CLM and HiCAP, whose action tables are fixed (§3).

### 1.3 Projection, temperature, loss, negatives

- **Joint space:** 512-d, cosine score.
- **Training temperature:** `logit_scale` = 2.6592 = ln(1/0.07), learned, **not clamped** in training (`T:209`, `T:416`). The PolaRiS checkpoint stores `logit_scale`. The Bridge checkpoint omits it because inference uses raw cosines.
- **Loss:** symmetric CE over the in-batch B×B logits, (image→action + action→image)/2 (`T:893–899`; paper Alg. 1 l.15–18).
- **Negatives:** **in-batch only.** There are no hard negatives, no false-negative mask, no near-miss mining and no prior or logit adjustment (code and paper §4.2: "treating other actions in the batch as implicit negatives"). The paper's lever for negatives is **batch size**: 2,048 → 8,192 improves retrieval (§5.1, numbers in the Fig. 5 image only), and the final model uses 32,768 (§5.2).
- **Optimiser in the code:** AdamW (torch default weight decay), lr **1e-6**, linear warm-up over 10 epochs then constant, grad clip 1.0, 50 epochs, batch 64 per GPU (`T:677`, `T:830–837`, `T:906`, `T:1408–1436`).
  - ⚠️ These are **code defaults, not the paper's final run.** The README's to-do list still says "Release verifier training pipeline".
- ⚠️ **The training validation split is sample-level, not trajectory-disjoint.** `torch.utils.data.random_split` runs over samples (`T:783–786`), so the in-training val loss shares trajectories with train. This is the REF-A I-JEPA leak class. The paper's *reported* retrieval accuracy uses held-out trajectories with "unseen environments and instructions" (§8.4), so the published evaluation is not affected. Only the code's internal val is.

### 1.4 Trainable parameter count (MEASURED, Appendix A)

| component (per ensemble member, Bridge) | params |
|---|---|
| vision attention pooler | 8,410,624 |
| text attention pooler | 8,410,624 |
| input projection 1024→512 | 524,800 |
| text-aware extraction (temperature only; the 576×1024 `pos_emb` is a fixed buffer) | 1 |
| **state side subtotal** | **17,346,049** |
| single-step action encoder 7→512 | 4,096 |
| trajectory transformer (4 layers) | 8,411,136 |
| **action side subtotal** | **8,415,232** |
| **trainable per member** | **25,761,281** |
| **× 3 members (deployed ensemble)** | **77,283,843** |

Cross-check: the stored floats including buffers come to 26,351,105 × 3 × 4 B = 316.2 MB, against a file size of 316,400,837 B. PolaRiS adds +262,144 (projection 1536 wide), +512 (D = 8), +15,872 (positional) and +1 (logit_scale) per member.

The paper's "**1B-parameter**" verifier (§5.2) counts the frozen SigLIP 2 backbone. The trainable part is about 26 M per member (MEASURED); its §8.4 attributes the 250M→1B scaling largely to the larger *text* encoder (280 M against 40 M for CLIP). CLM's heads are **≈20 M state + ≈20 M action**, and HiCAP's NC10 CLM-pure state side is **22.3 M** (INHERITED, `sec_protocol.tex:18`). **CoVer's 17.3 M state side, built from two attention poolers, is the same size class.**

### 1.5 Training data and steps

| item | paper (PUBLISHED-PAPER) | code |
|---|---|---|
| data | Bridge V2 augmented with **16×** synthetic instructions per original. GPT-4o generated **128 variants per instruction**, which were embedded with Qwen3-Embedding-0.6B and k-means-curated into 8×/16×/32×/64× subsets (§8.4) | `BridgeDataset` over a normalised JSON of (image file, instruction id, action-history id) (`T:24–179`) |
| scale | "20 million offline samples" (§1) | — |
| batch, steps, hardware | batch **32,768**, **2k steps**, **8 × H200** (§5.2) | CLI default 64 per GPU (`T:1409`) |
| ensemble | 3 for deployment, trained identically except for the seed (§5.1, §5.2) | the checkpoint holds 3 `ensemble_components` (MEASURED) |
| policy-augmentation baseline | π0 fine-tuned on OXE-rephrase-augmented Bridge (Fang et al.) | INT-ACT configs |

### 1.6 Candidates at test time

| item | fact | evidence |
|---|---|---|
| VLA | **π0** fine-tuned on Bridge (`juexzz/INTACT-pi0-finetune-bridge`, or `…-rephrase-bridge` for the "π0 (rephrase)" arm), with chunk_size = n_action_steps = **4** | `R:65`, `R:149–155`; `INT-ACT/config/models/pi0_finetune_bridge.json:35–36`. PolaRiS uses **π0.5** (`pi05_droid_jointpos_polaris`; cover-vla-polaris README l.99) |
| diversity source | the flow-matching initial noise, with **`noise_std = 1.0`** | `R:186`, `R:322`; lerobot_custom `modeling_pi0.py:264`, `:502–505` |
| K × M | **K = 8 prompts × M = 5 samples = 40**, in one batched π0 call | `test_pi.sh` (`--lang_rephrase_num 8 --policy_batch_inference_size 5`); `R:297–322`; paper §5.3 |
| prompts | slot 1 is the *currently active* instruction. Slots 2–8 come from a precomputed list of **ERT red-team-derived rephrases** (`ert_rephrases`, 33 per task, first 7 used) | `prompt_utils.py:10–27`; `R:209`; `simpler_rephrased_final_eval_vlm.json` |
| rephraser | **GPT-4o, temperature 0.8**, given the instruction **and the task's initial frame**. Prompted to describe the scene, reinterpret the instruction and list noun/verb/adjective substitutes, then reword. Produced once per task, offline ("boot time"; about 11 s for 8 rephrases, §8.9) | `bridge_verifier/lang_transform_vlm.py:47–99`, `:115`, `:148`; `generate_simpler_rephrases_vlm.py:18`, `:250`. PolaRiS: **Gemini** (`gemini-3-pro-preview`, env fallback `gemini-2.0-flash`), on-the-fly on a cache miss and then persisted (`PS:194–237`, `PS:873–875`) |
| evaluated instruction | the episode starts from the **red-team** instruction, e.g. "Set the green squash straight on the patterned cloth.", not the template ("put the zucchini on the towel") | `R:217`; JSON `original` field; paper §5.3, §8.8 |

### 1.7 Selection

**Paper (Alg. 2, §4.3).** Every one of the K×M candidates is scored against the **user** instruction, $s_{k,j}=\mathcal V_\theta(o_t,h_t,l',a_{k,j})$. The prompt chosen is $k^\*=\arg\max_k \frac1M\sum_j s_{k,j}$ (a **mean**, which measures the quality of the distribution a prompt induces), and the action chosen is $j^\*=\arg\max_j s_{k^\*,j}$.

**Code (Bridge, `R:327–407`).**
1. **Stage-1 gate.** Score **only candidate 0** (the first sample under the active instruction). If its cosine is **≥ 0.1**, execute it and stop. No selection happens (`R:342–353`).
2. **Stage 2.** Otherwise score all 40 against the **active instruction** (`R:355–361`, `instructions=[task_description]*batch_size`). Take the group mean, then the argmax within the winning group, exactly as in Alg. 2 (`E:427–436`).
3. **Gripper majority vote** within the winning prompt group (`R:372–390`). This is not in the paper.
4. **Sticky instruction.** `task_description = max_instruction` (`R:407`). The *selected rephrase* becomes both the next chunk's slot-1 prompt **and the text the verifier scores against.** The paper says scoring is always against the user instruction (§4.3; §8.8: "the similarity score is calculated between generated actions from rephrases and the original user instruction").

**Code (PolaRiS, `PS:576–674`).** Stage 1 scores the **M actions of the current prompt**, with a threshold of **0.45** (`PS:598`). Stage 2 is the full K×M selection against `current_instruction`, which is the sticky selected instruction (`PS:306`, `PS:360`, `PS:610–616`).

**Ensemble fusion (`E:282–293`, `E:403–411`).** Each member's state embeddings and action embeddings are **averaged across members separately, then re-normalised**, and only then dotted. The score therefore contains cross terms $\langle \mathbf f_i,\mathbf a_j\rangle$ for i ≠ j between independently seeded members. With 3 members that is 6 of 9 terms, and they are meaningful only to the extent the members' spaces are aligned. The paper (§5.1) reports that this improves retrieval from 1 to 8 members (numbers in the image only). It does not compare against averaging the scores.

### 1.8 Latency, rate, caching, asynchrony

| item | fact | evidence |
|---|---|---|
| when verification runs | at **every chunk boundary** (`real_step % n_action_steps == 0`). In between, the selected chunk's remaining steps are popped from a queue, open-loop | `R:320–327`, `R:392–420` |
| Bridge rate | control **5 Hz** (sim 500 Hz); chunk 4, so one decision per **0.8 s of simulated time**. 150-step episodes = 30 s | SimplerEnv `put_on_in_scene.py:170–171`; `R:92`, `R:247` |
| PolaRiS rate | control **15 Hz** (dt 1/120 × decimation 8); server default `n_action_steps` **16**, so ≈1.07 s per decision | cover-vla-polaris `src/polaris/environments/droid_cfg.py:357–358`; `PS:100`, `PS:1098` |
| wall clock (paper, RTX 5090) | π0 batch forward **344 ms (bs 1) → 748 ms (bs 32)**; CoVer action encoder **7–8 ms**, roughly constant; total **< base + 10 ms**; **16 candidates ≈ 453 ms (≈2.2 Hz)**; bs 32 → **756 ms ≈ 1.3 Hz**, "works for most quasi-static manipulation" | PUBLISHED-PAPER §5.6, §8.7. Tables 2 and 5 themselves are images; these are the in-text values. **The latency at K×M = 40 is not in the text** |
| claimed overlap | "the image-text embedding [is] computed in parallel with the forward pass of the base robot policy" (§5.6) | ⚠️ **not in the released code.** `R:320–361` calls π0 and then the verifier sequentially |
| caching | the rephrase **list** is computed at boot time and cached (JSON on disk). The paper's Alg. 2 l.2 also says "cache embeddings". ⚠️ The released Bridge code re-encodes the single active instruction and the image at every verification (`E:332–347`). Its rephrases go only to the **policy**, so no rephrase embeddings are needed by the verifier | `U:292–307`; `PS:239–268` |
| multi-rate / asynchrony | **none.** The only separation in time is boot time (VLM rephrasing, once per episode) versus run time (π0 plus verifier per chunk). Rephrases are not regenerated mid-rollout, by design (§4.3: "user intent is typically consistent throughout an episode") | paper §4.3, §8.9 |

---

## 2. Results, with their exact numbers and settings (priority 2)

**Evaluation protocol** (PUBLISHED-PAPER §5.3 and §8; code `test_pi.sh`):
- **Every** evaluation uses **ERT red-team instructions**: adversarial paraphrases built to break π0 (§8.8, Table 8).
- K = 8 rephrases × M = 5 samples.
- SIMPLER WidowX: **100 trials per task** (`test_pi.sh --num_trials_per_task 100`).
- "±" is a standard deviation. The paper does not state the seed count for SIMPLER; PolaRiS uses 50 episodes × 3 seeds (Table 1 caption).

### 2.1 SIMPLER, success rate %, Table 3 (in the text and in the README)

| arm | Carrot | Eggplant | Spoon | Block | **ID avg** | Redbull | Zucchini | Tennis | **OOD avg** |
|---|---|---|---|---|---|---|---|---|---|
| π0 | 48±4 | 74±3 | 27±4 | 17±1 | **41.5** | 6±1 | 30±3 | 53±5 | **29.7** |
| π0 w/ Inst. Aug. (policy scaling) | 48±4 | 68±4 | 36±2 | 24±4 | **44.0** | 29±3 | 42±3 | 75±9 | **48.7** |
| π0 w/ random rephrase | 49±5 | 78±3 | 41±1 | 1±1 | **42.3** | 1±1 | 7±2 | 78±3 | **28.7** |
| RoboMonkey (7B verifier, re-based on π0) | 0±0 | 13±3 | 10±1 | 7±2 | **7.5** | 19±3 | 10±5 | 45±1 | **24.7** |
| π0 + CoVer | 48±4 | 89±8 | 40±6 | 51±4 | **57.0** | 51±3 | 41±1 | 91±3 | **61.0** |
| π0 (rephrase) + CoVer | 52±8 | 95±2 | 59±5 | 56±0 | **65.5** | 46±3 | 55±6 | 85±1 | **62.0** |

**Reconciling the headline numbers** (MEASURED arithmetic on the table):

| claim in the paper | which arms are compared | value from the table | matches? |
|---|---|---|---|
| abstract: "**22 %** ID / **13 %** OOD" | π0 (rephrase) + CoVer vs π0 w/ Inst. Aug. | **+21.5 / +13.3** points | ✓ |
| §5.4: "+16 % ID / +31 % OOD" | π0 + CoVer vs π0 | +15.5 / +31.3 | ✓ |
| §5.4: "π0 + CoVer beats policy scaling by **15 %** ID / 12 % OOD" | π0 + CoVer vs π0 w/ Inst. Aug. | **+13.0** / +12.3 | ⚠️ the "15 %" does **not** match the table's 13.0. This is an internal inconsistency in the paper |
| contribution list: "**18 %** on SIMPLER" | π0 (rephrase) + CoVer vs π0 w/ Inst. Aug., **7-task mean** (64.0 vs 46.0) | +18.0 | ✓ (this is my reconstruction; the paper does not say which comparison it means) |

All of these gains are in **absolute percentage points of success**.

**How to read the SIMPLER table.**
- The **random-rephrase** arm (42.3 / 28.7) shows that a rephrase alone is a gamble: Block 1 %, Redbull 1 %, against Eggplant 78 %. **Selection** among rephrases carries the gain.
- **RoboMonkey collapses below base π0.** The paper attributes this to distribution shift and to step-level verification breaking up chunks (§8.2). It is therefore a weak "strongest prior" comparator.
- **Gain sizes vary widely by task.** The largest gains are on object-grounding failures (Redbull: the "red" in Redbull pulls π0 to the Coke can, §8.1.1; Block). On Carrot the gain is 0.
- **The rephraser injects scene grounding.** GPT-4o sees the initial frame and writes "blue can", "green-handled spoon" (Table 8). Part of the gain is therefore a large VLM restoring grounding that the red-team phrasing removed. The compute comparison in Fig. 1 counts π0 plus verifier training, **not** GPT-4o.

### 2.2 Real world, WidowX, two tasks (PUBLISHED-PAPER §5.5, Fig. 8)

"pepto bismol on plate" and "redbull on plate": **+30 and +60 points**, **45 points average absolute** over π0 (rephrase). Base π0 "often failed to initiate motion … resulting in 0 % success". **The number of trials is not in the text** (the figure is an image).

### 2.3 PolaRiS, π0.5 base, 50 episodes × 3 seeds

The Table 1 caption gives the averages: **+13.9 points task progress, +9.3 points success.** Table 1 itself is an image. The per-task values below come from a **commented-out block in the repo README** (PUBLISHED-CODE, README; this is not the paper table). They reproduce the caption's averages exactly (MEASURED: (48.4+33.1+38.3)/3 = 39.9; (70.4+44.3+47.0)/3 = 53.9; (10.7+0+0.7)/3 = 3.8; (33.3+0.7+5.3)/3 = 13.1).

| task | π0.5 progress | π0.5 success | +CoVer progress | +CoVer success |
|---|---|---|---|---|
| PanClean | 48.4±1.9 | 10.7±0.9 | 70.4±4.0 | 33.3±6.6 |
| BlockStack | 33.1±1.3 | 0.0±0.0 | 44.3±2.5 | 0.7±0.9 |
| FoodBussing | 38.3±2.4 | 0.7±0.9 | 47.0±4.1 | 5.3±1.9 |
| **avg** | **40.0** | **3.8** | **53.9** | **13.1** |

### 2.4 Verifier-level results (PUBLISHED-PAPER)

- **Binary discrimination**, ground-truth versus *randomly sampled* actions: precision **0.765**, recall **0.780**, F1 **0.772** (§8.5.1). Random negatives are the easiest possible negatives, so this is a weak test.
- **Best-of-N regression error:** RMSE of the selected action against ground truth on held-out trajectories, **−11 % at N = 16 vs N = 1** (§8.5.3; Table 6 is an image).
- **Verifier scaling** (§5.1, Fig. 5): top-1 retrieval in a **fixed pool of 64** (held-out trajectories; GPT-4o-rephrased instructions, §8.4) rises monotonically with:
  - synthetic instructions 8× → 64×;
  - parameters 250 M → 1 B (attributed mainly to the text encoder, §8.4);
  - batch 2,048 → 8,192;
  - training epochs;
  - ensemble size 1 → 8.

  **Every value is in the figure image only.**
- **Verifier score over time** (§8.5.2, Fig. 10): successful episodes peak at the approach and at completion, while failed episodes decline steadily. The paper proposes this as a runtime monitor. The claim is qualitative.

### 2.5 Test-time scaling-law claims: what is quotable

- **The model** (§3, Fig. 2 caption): a power law $\log e \approx \log a + b\log k$, where $e$ is the **oracle** action error (NRMSE of the **best** of $k$ candidates against the ground-truth action), fitted per sampling strategy.
- **The data:** **n = 1,000** (s, a, I) tuples sampled uniformly from Bridge V2. The four strategies are repeated sampling, Gaussian perturbation, instruction rephrasing, and hybrid (rephrase × repeat).
- **Text-only claims:** (1) rephrasing gives lower oracle error than repeated sampling or Gaussian perturbation; (2) the hybrid is lower still; (3) error versus inference FLOPs "follows an exponentiated power law".
- ⛔ **No exponent, fit window or R² appears in the text.** The values of $b$ are only in the Fig. 2 image. **Per CLAUDE.md, no exponent from this paper is quotable here**, and none is quoted.
- ⚠️ **Even if the exponents could be read, this is an *oracle* (min-over-k) law.** It measures **coverage or diversity** of the proposal set, not what a verifier recovers. This is exactly the quantity the registry warns about: REF-C-XL's oracle-in-fan **0.1640** against a pick of **0.4714** is "~92 % irreducible … a minimum over 256 candidates scored against ONE realised future" (INHERITED, registry l.1347–1360). CoVer's own *verifier-selected* error improvement is the −11 % above, not the slope of Fig. 2.

---

## 3. Mapping to HiCAP and REF-F, element by element (priority 3)

### 3.0 Element table

| CoVer-VLA element | HiCAP / REF-F | status |
|---|---|---|
| frozen backbone, small trainable heads, symmetric in-batch InfoNCE | HiCAP stage 1 (CLM recipe; `sec_method.tex` l.28, l.106); REF-F `L_nce` | **HAVE** |
| learned temperature, init 0.07 | τ init 1/0.07, **clamped at 100** (`sec_method.tex` l.96) | **HAVE.** CoVer clamps only the text-aware temperature, not the logit scale |
| backbone: dual image–text encoder (SigLIP 2-L), patch tokens from the last block's attention output | a generative VLM (Cosmos / Qwen3-VL family), visual tokens at 3 LM depths pooled to ≤128, plus the last token | **DIFFERENT.** Ours reads deeper, multi-depth features of a generative LM; theirs reads one shallow-ish ViT tap |
| text-conditioned patch attention (instruction tokens as queries) | none. The nav token enters the fusion transformer as one token | **LACK** (see 3c) |
| single-query 4-layer attention pooling (17.3 M state side) | fusion transformer, 6 layers, d 512, ≈19 M, with level queries, motion, ego and nav tokens; NC10 CLM-pure = last-token MLP (22.3 M) | **DIFFERENT.** The aggregator design that lies between those two arms is not tested (P2) |
| action tower over history + chunk, re-embedded every decision | **cached** action tables (text-node towers on frozen definitions; numeric path and profile towers); ego history on the **state** side | **DIFFERENT by design.** Caching is HiCAP's latency argument (§3e). CoVer's choice forfeits it |
| in-batch negatives only, no prior correction, raw cosine argmax | level-wide negatives, VADv2 soft targets, false-negative mask, stage-2 mined near-misses, log-prior β correction (H-F4, H-HC4) | **HAVE MORE.** CoVer is evidence that a selector *works* without these, not that they are unnecessary; it never ablates them |
| generate-then-verify (40 π0 samples per chunk) | retrieve from a fixed vocabulary, then refinement head, then S4 cross-attention re-rank of K ≤ 32. REF-F **H-F3** = verifier mode over REF-C-XL's 256 fan | **PARTLY HAVE** (H-F3 is pre-registered, not run) |
| hierarchical prompt-mean then action-argmax | strategic → tactical (lat×lon) → path×profile beam with marginal critics | **DIFFERENT** (see 3b) |
| boot-time VLM rephrasing of the instruction | none (no instruction text; strategic family UNAVAILABLE on PhysicalAI) | **LACK, and gated** (3b) |
| seed ensemble with embedding averaging | 3 seeds per arm trained for statistics, **not ensembled at inference** | **LACK, and free** (P3) |
| confidence-gated expansion (stage 1 → stage 2) | System-2 only when the top-two margin < m_min or the imagination heads disagree; S4 escalates K 16→32 on a small S1 margin; conformal keep-sets | **HAVE**, with calibrated thresholds where CoVer hand-sets cosines of 0.1 and 0.45 |
| sticky instruction (scores against its own previous selection) | — | **must not adopt** (E3) |
| chunk-boundary verification at ~1 Hz, open-loop between decisions | 2 Hz VLM refresh, 10 Hz fusion and trajectory readout, levels at 0.5/2/10 Hz, hysteresis R5 | **DIFFERENT regime** (3e) |

### 3a. "Generate-then-verify" versus selecting from a fixed vocabulary

- **What CoVer actually demonstrates** is closed-loop success gains from *selecting among a generator's modes* when the generator's failure is a **semantic mode error**: the wrong object, or failing to start. At the regression level it buys −11 % RMSE at N = 16.
- **What the programme has already measured on its own generator** (INHERITED, registry):
  - REF-C-XL's 256-fan has an oracle of **0.1640** against a selected **0.4714** on val-40 / 881 windows (l.1347–1348).
  - A learned re-scorer recovered **at most 8.4 % of that gap across 47 trained arms**. That figure is a prose note, not a results JSON (l.1355–1360, l.712).
  - Cost re-ranking recovered **0.0 %** (l.1365–1367).
  - **Two independent systems therefore agree on the order: ~10 % of a regression gap.**
- **Consequence for the CoVer row of HiCAP's argument.** "Verification beats policy scaling" should be **not** read as "the ADE oracle gap is a lever". If it transfers, it transfers at the **TACTICAL** level: choosing the right manoeuvre cell among modes. That is where a scalar ADE cannot see the error, by the programme's own BINDING four-family rule.
- **HiCAP's fixed vocabulary removes the generator from the loop.** It sees all modes by construction (FPS-built, factored lat×lon) and needs no K×M sampling to obtain diversity. That is CoVer's "diversity via conditioning, not noise" insight (§2.5) achieved structurally.
- **Generated candidates remain useful in two roles:**
  - as **open candidates** that the vocabulary lacks; HiCAP's refinement head plays this role;
  - as a **second opinion** in verifier mode (REF-F H-F3).

  For generated candidates the action tower runs per candidate, which is CoVer's regime. HiCAP's numeric towers total about **1.1 M parameters** (INHERITED, `sec_method.tex` l.43), so that is cheap.
- **Alpamayo 2 samples are not usable now** (E2: no coordinates are stored).
- **Bottom line:** for HiCAP, verification is a **tactical-mode instrument**, not a gap-recovery instrument.

### 3b. Prompt × action hierarchy versus strategic / tactical / trajectory, and whether rephrased goals are admissible

**The structural analogy.**
- CoVer's high level chooses a **conditioning** variable (the prompt fed to the generator), scored by the **mean** quality of the distribution it induces. Its low level chooses a **sample** within that distribution.
- HiCAP's high levels choose **action-space nodes** (strategic token, tactical cell) whose subtrees bound the children (a hard beam). The node is scored by a marginal critic, which is a log-sum-exp-like aggregate over its subtree (Proposition `prop:marginal`).
- **CoVer's group mean is the cruder version of HiCAP's marginal critic.** A mean over 5 samples equals "average child", and HiCAP's own design text warns that this "scores the *average* child and hides a narrow peak inside a large cell" (`sec_method.tex` l.92). Nothing should be adopted here. CoVer is an instance of the failure that HiCAP's design already avoids.

**Could rephrased navigation or goal instructions play the role of CoVer's rephrased prompts?** Only under four conditions, and today it is gated:
1. **There is no instruction text to rephrase.** PhysicalAI-AV carries no route, goal or map (CLAUDE.md rule 2, settled at five probes). The nav command is a 4-way categorical that is **oracle-perfect** in the corpus (`sec_method.tex` R3). Rephrasing a categorical means nothing to a model that consumes a categorical token. The STRATEGIC family is UNAVAILABLE (n = 0) until AlpaSim or a route-bearing corpus is used.
2. **What does transfer is *conditioning diversity*,** used as a candidate generator. The goal head predicts goal points (x, y, t_reach) at 2/4/6 s (`sec_method.tex` l.108). Sampling **several predicted goal points** and composing candidates under each one is the CoVer-hybrid analogue: diversity from conditioning, not noise (P4).
3. **BINDING admissibility (Sayed, 2026-08-03).** The goal input must not carry the situation classifier's output.
   - **A text-only rephraser of a route string is admissible.**
   - **An *image-conditioned* VLM rephraser, CoVer's choice, is NOT automatically admissible.** CoVer's prompt makes the VLM "describe the scene … then reinterpret the instruction" (`lang_transform_vlm.py:75–78`). In driving, that scene description *is* situation classification ("at the roundabout", "behind the stopped bus"). A rephrase containing those words carries a situation label into the goal path. It may not be our classifier's output, but it is a parallel classifier's output, and **the attribution failure is the same.**
   - Admissible design: (i) text-only rephrasing; or (ii) an image-conditioned rephraser with a **closed lexical filter** that removes situation-class vocabulary, **plus a leak probe** (a linear probe from the goal embedding to the sitclf label must not beat a shuffled-label control). Run the check *"could this have been computed from the situation classifier's output?"* on every rephrase set.
   - The vision-only rule is satisfied either way (image only, no ego), **provided the rephraser receives no ego state.**
4. **Anti-echo.** CoVer's code scores against the instruction its own selector chose (§1.7). In HiCAP terms, that is a tactical verifier scoring against the strategic token that the same pipeline produced, or a goal echo. Any HiCAP verifier must score against an input it did not choose, and must report a sticky-off control (E3). The NC6 strategic-token-null and goal-swap tests are the right instruments; add "verifier-reference = own previous selection" to NC6.

**Where language rephrasing does belong in HiCAP now** is the **System-2 verifier**. It renders the top K₂ ≤ 8 survivors as short text ("lat=LANE_KEEP, lon=BRAKE_TO, target 6 m/s"; `sec_method.tex` l.142). CoVer's random-rephrase arm shows a language-conditioned model's behaviour swinging from **1 % to 78 %** across paraphrases (Table 3). A text-rendered verifier is therefore exposed to the template's wording. P5 measures that exposure; it does not assume it.

### 3c. Attention-pooling state encoder versus our fusion transformer and NC10

**Three state-side designs exist, all about 17–22 M parameters:**

| arm | state side | size |
|---|---|---|
| (i) NC10 CLM-pure | LN + MLP on the frozen **last token** | 22.3 M |
| (ii) CoVer | text-conditioned patch attention + **single-query 4-layer attention pooling** over patch tokens, no other inputs | 17.3 M |
| (iii) HiCAP | **6-layer fusion transformer** over ≤128 VLM tokens × age embedding, plus 16 motion tokens, ego, nav and level queries | ≈19 M fusion + 4.6 M projector + 0.6 M motion + 0.9 M tokens (INHERITED, `sec_method.tex` l.43) |

**NC10 compares (i) with (iii)** and attributes the whole difference to "fusion transformer, motion adapter and ego/nav/route tokens" (26.4 M).

**The confound.** That difference mixes two factors:
- **the readout**: last-token vector versus learned aggregation over token sets;
- **the inputs**: VLM only, versus VLM + motion + ego + nav.

CoVer is a ready-made design for the missing cell, **learned aggregator, VLM tokens only**. That is P2 (NC10-AP).

**Two smaller points.**
- **Which layer to tap.** CoVer taps the last ViT block's **attention-sublayer output, pre-residual** (`T:271–274`). That is an unusual tap. HiCAP's three-depth LM taps are better motivated and should stay. CoVer is not evidence for its tap; no ablation was run.
- **The instruction-conditioned query.** CoVer's text-aware extraction (instruction tokens querying patches) is the one genuinely new state-side idea for HiCAP. The analogue is a *nav/goal*-conditioned visual readout. It is admissible only with a nav or predicted-goal source (not sitclf); on PhysicalAI the nav is oracle, so only the nav-zero row is deployment-honest. It is a P2 sub-arm, not a default.

### 3d. Verification scaling versus policy scaling as a programme lever

**CoVer's evidence:**
- Same data, π0 + CoVer **+13.0 / +12.3 points** over the policy-scaled π0 (Table 3).
- The two are complementary: rephrase-trained π0 + CoVer is best.
- The verifier needs **~10³× less backward compute**, because the backbone is frozen (§8.6: backward ≈1.0×10⁹ against forward ≈3.3×10¹¹ FLOPs per sample).

**Our evidence:** on open-loop ADE, selection over a strong fan recovered ≤ 8.4 % (INHERITED), and the registry declared "**selection is no longer the productive lever on REF-C**" (l.1361–1362).

**Reconciliation, HYPOTHESIS.**
- CoVer's lever is **semantic alignment under adversarial instructions** in closed loop. Our measured null is **regression accuracy against one realised future** in open loop.
- **These are different questions. Neither result refutes the other.**
- For TanitAD the CoVer claim becomes testable in two places:
  - on the **TACTICAL** family, where a mode error (brake versus keep, lane-change versus keep) is the analogue of grasping the wrong can;
  - in **closed loop** (REF-F H-F8 on AlpaSim), where success is the metric, as in CoVer.
- **Programme lever, the claim to test:** *verification compute should be spent where the decision is discrete and consequential* (tactical cadence, low-margin ticks). This is already HiCAP's System-2 policy. CoVer adds a *trained contrastive* verifier as an alternative to the VLM-weight-pass verifier, at about 1/10 of the cost (HYPOTHESIS: a ~26 M head versus a 58.6 ms bf16 8 B weight pass, ESTIMATED in `sec_method.tex` l.143).
- **Do not fund "scale verification" as an ADE lever without P1's TACTICAL result.**

### 3e. Frequency and latency regime versus HiCAP's 2 Hz refresh + 10 Hz fast path

| | CoVer-VLA | HiCAP |
|---|---|---|
| decision rate | one per chunk: 1.25 Hz sim (Bridge), ≈0.94 Hz (PolaRiS); wall clock 1.3–2.2 Hz | trajectory 10 Hz; tactical 2 Hz; strategic 0.5 Hz |
| between decisions | **open-loop** chunk replay | fusion and trajectory readout re-evaluated every 100 ms on aged VLM tokens plus fresh motion and ego tokens |
| per-decision critical path | π0 K×M batch forward (344–748 ms) + verifier action tower (7–8 ms) | cached-table dot products + S4 re-rank, ≈0.34 GMAC at K = 16 (INHERITED, ESTIMATED); REF-F target p95 ≤ 50 ms on Thor (H-F5) |
| expensive model | in the loop at every decision (π0) | VLM off the critical path (asynchronous 2 Hz refresh); System-2 only on uncertain tactical ticks |
| what is cached | the rephrase list (boot time) | VLM token sets (2 Hz), action embedding tables (permanent), S2 survivors (5 ticks) |
| asynchrony / multi-rate | none | yes (three level rates plus token aging) |

- **CoVer's regime is about an order of magnitude slower than HiCAP's fast path.** π0's 344 ms at batch 1 already equals 3.4 HiCAP ticks, on a desktop GPU (ESTIMATED arithmetic; Thor is slower).
- **What does transfer is the *decoupling* argument** (§5.6): state embedding independent of candidates, so per-candidate cost is only the action tower. HiCAP goes further: with a fixed vocabulary the action tower costs **zero** at run time.
- **Placement:** a CoVer-style verifier over *generated* candidates belongs **at tactical cadence (2 Hz) or slower**, behind the margin gate, exactly where HiCAP places System 2. **Never in the 10 Hz loop.**

---

## 4. Concrete, testable proposals (priority 4) — all HYPOTHESIS

Every proposal below follows the programme's rules:
- paired episode-cluster bootstrap (B = 2000), full-set means;
- all four metric families reported (STRATEGIC reported UNAVAILABLE with n = 0 on PhysicalAI);
- nothing tuned on val; τ, β and thresholds fitted on a held-out 10 % of train episodes;
- parity corpus `physicalai-train-e438721ae894` / `f09e44db`.

### P1. A CoVer-style contrastive verifier over REF-C-XL's fan, judged on TACTICAL (extends REF-F H-F3)

- **Design.**
  - State side: REF-F's frozen state head.
  - Action side: a CoVer action tower, i.e. `Linear` → 4-layer transformer **with positional encoding** (the PolaRiS fix; Bridge's lack of it is a defect) → masked mean, over [admissible ego-history steps ⊕ the candidate's 20 steps].
  - Training: symmetric in-batch InfoNCE **on generated fans**, with the ground-truth future as positive (CoVer's recipe); 3-seed ensemble; selection by group-mean over **tactical cells** (CoVer's prompt-level rule, applied with cells in place of prompts), then argmax within the cell.
  - Compare against REF-C's own pick, the Δ2 expected-cost regressor and REF-F's dot product.
- **Admissibility.** The ego history in the action tower must sit within the PI's ego ruling (velocity, longitudinal acceleration and yaw rate at cycle time; REF-F's pre-registered 8-step speed history is the precedent). Run NC4 (ego-only copycat).
- **Cheapest discriminating experiment.** Val-40 and val-600 on REF-C-XL fans that H-F3 already requires. Fans for train are needed for training; **0 extra GPU-days if H-F3's fan cache is built anyway.** The primary metric is **TACTICAL**: the agreement between selected and executed (lat, lon) cell, macro-F1 over populated cells, paired against REF-C's pick. ADE gap recovery is reported but **cannot decide** (lottery statistic).
- **Outcome A.** TACTICAL separated-better **and** ADE non-inferior (upper CI95 < +0.02 m). Then CoVer's thesis transfers at the mode level: adopt the contrastive verifier as HiCAP's **cheap System-2** for generated or open candidates, behind the margin gate at 2 Hz.
- **Outcome B.** TACTICAL not separated, **or** ADE separated-worse. Then verification over this fan carries no mode signal on PhysicalAI, which is consistent with the registry's ≤ 8.4 %. The CoVer lever is tested only in closed loop (H-F8, n ≥ 30 scenes) and is not funded further open-loop.

### P2. NC10-AP: CoVer's aggregator as a third NC10 arm (attention pooling, VLM tokens only)

- **Design.**
  - `clm_pure.py`'s last-token MLP is replaced by **one single-query attention pooler per level**: 4 cross-attention layers, d 512, 8 heads, mlp 512, as in `M:76–112`, over HiCAP's cached ≤128 tokens × 3 depths.
  - Each pooler is about 8.4 M at kdim 1024 (MEASURED shape-equivalent; it scales with d_v).
  - No motion, ego, nav, route or age tokens.
  - Size-matched to CLM-pure (22.3 M) by width, or reported with its count.
  - **Sub-arm NC10-AP-q:** CoVer's *query-conditioned* patch attention with the **nav** token's text as query. Nav-fed and nav-zero rows; admissible because nav is not a sitclf output.
- **Cheapest discriminating experiment.** A head-only arm on the **existing** HiCAP token cache, 3 seeds, paired against CLM-pure and against HiCAP on the same windows. This costs the same as NC10 itself.
- **Outcome A.** NC10-AP non-inferior to HiCAP in all four families. The fusion transformer's value is **the aggregator, not the added inputs**. Prefer NC10-AP (fewer parts), and re-examine whether the motion adapter, ego and nav tokens earn their ≈7 M.
- **Outcome B.** NC10-AP ≈ CLM-pure (neither separated in any family) while HiCAP beats both. The aggregator buys nothing over the last token, so HiCAP's gain comes from the **inputs**. Keep the fusion transformer, and fix `sec_method.tex:23`'s aggregator claim (E1b) with this measurement as its citation.
- **Mixed** results are reported per family and never pooled.

### P3. Seed ensemble at zero training cost: embedding mean (CoVer) versus score mean

- **Design.** Every HiCAP and REF-F arm already trains 3 seeds for the noise floor. At inference, compare four scorers on the same windows:
  - (a) each single seed;
  - (b) CoVer's **embedding-mean**: average the 3 state embeddings and the 3 action embeddings separately, re-normalise, dot (`E:403–411`);
  - (c) **score-mean**: average the 3 per-seed calibrated logits;
  - (d) the per-seed rank-vote.
- **Why it discriminates.** Under (b), 6 of the 9 terms are cross-seed dot products. If the seeds' 512-d spaces are not aligned (rotation freedom in InfoNCE), those terms are noise and (b) is no better than (a). If the shared frozen backbone and cached text towers align them, (b) ≈ (c).
- **Cost:** evaluation only. Zero training.
- **Outcome A.** (c) or (b) separated-better than the best single seed in ≥ 1 family and non-inferior elsewhere. Then ship a 3-member head ensemble; the heads are small, so the 3× head cost is ≈1 ms class (ESTIMATED). If (b) ≈ (c), the spaces are aligned, which is a reportable fact about InfoNCE plus a frozen backbone.
- **Outcome B.** No ensemble separated from single seeds. Then CoVer's ensemble gain (§5.1, figure only) does not transfer at our scale; report the seed spread and drop it. Also report whether (b) < (c), i.e. whether CoVer's cross terms hurt.
- **Protocol caveat.** The headline stays "mean of 3 seeds" per the pre-registration. The ensemble is a **separate, pre-registered arm**, and its interval pools episodes, never seeds.

### P4. Conditioning-diverse candidate sets versus noise-diverse sets, at matched K (CoVer §3 hybrid, done correctly)

- **Design.** Build candidate sets of equal size K = 64 from REF-C-XL in three ways, plus one control:
  - (i) **noise only**: 64 diffusion samples or anchors under the logged nav;
  - (ii) **conditioning only**: 16 under each of the 4 nav commands (nav-swap), and separately under predicted-goal-point perturbations once the goal head exists;
  - (iii) **hybrid**: 4 conditionings × 16 samples;
  - control: HiCAP's FPS vocabulary pruned to 64.
- **Metric: NOT min-ADE.** Min-ADE is the oracle lottery CoVer's Fig. 2 uses. Use **mode coverage**: the fraction of windows whose *executed* tactical cell appears in the set; plus **cell entropy**; plus selected-ADE under one fixed selector.
- **Admissibility.** Nav variants are not sitclf outputs. Predicted goal points come from the goal head and must pass the "could this be computed from sitclf output?" check (the goal and tactical heads share s₂, so run the goal-shuffle control).
- **Cost:** inference only on REF-C-XL, val-600.
- **Outcome A.** Conditioning or hybrid separated-better on mode coverage at matched K. Then CoVer's diversity finding transfers: HiCAP's S2 composition and any generator arm should draw candidates across conditionings, and the FPS vocabulary is checked against it.
- **Outcome B.** No separation, or the FPS vocabulary ≥ all generated sets. Then the fixed factored vocabulary already supplies the diversity CoVer obtains by rephrasing, and generated candidates add nothing beyond refinement.

### P5. System-2 paraphrase-stability instrument (for when System 2 exists)

- **Design.** For each uncertain tick, render the K₂ ≤ 8 survivors with **8 paraphrases of the System-2 text template** (text-only paraphrasing, no scene words, so admissible under BINDING 2026-08-03). Record:
  - the **verdict-flip rate**: the fraction of ticks whose System-2 argmax changes across paraphrases;
  - TACTICAL accuracy per paraphrase.
- **Cost:** System-2 inference only. 0 training.
- **Outcome A.** Flip rate below a pre-registered 10 %, and per-paraphrase TACTICAL spread not separated. A single template is fine.
- **Outcome B.** Flip rate ≥ 10 %, or a separated spread. Then System 2 must score the **mean over paraphrases**, CoVer's group-mean rule applied to the verifier's own wording, at K₂ × 8 cost (budget re-check against `sec_method.tex` l.143). This is also a reportable robustness defect.

### Not proposed, with reasons

- **CoVer's sticky instruction.** It is an echo loop (E3).
- **History on the action side for HiCAP's vocabulary.** It destroys action-table caching, HiCAP's latency argument; history stays on the state side. It is allowed only for generated candidates (P1).
- **Online VLM rephrasing in the loop.** CoVer itself rejects it (11 s per call, §8.9).

---

## Appendix A — checkpoint inspector (MEASURED provenance; run with `python3 -I`, scratch only)

This restricted unpickler returns local stubs for exactly three globals (`collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch._utils._rebuild_parameter`) plus the storage and dtype names. **Any other global raises.** It imports neither torch nor repo code, and it reads only `data.pkl` from the zip, never the tensor bytes.

```python
import sys, zipfile, pickle, collections, math
class Storage:
    def __init__(s, dtype, key, numel): s.dtype, s.key, s.numel = dtype, key, numel
class T:
    def __init__(s, storage, offset, size, stride, *a): s.dtype = storage.dtype; s.shape = tuple(size)
    def __repr__(s): return f"T{s.shape}:{s.dtype}"
def rebuild_tensor_v2(storage, offset, size, stride, *a): return T(storage, offset, size, stride)
def rebuild_parameter(data, requires_grad, *a): return data
class DT:
    def __init__(s, name): s.name = name
    def __repr__(s): return s.name
ALLOWED = {("collections", "OrderedDict"): collections.OrderedDict,
           ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor_v2,
           ("torch._utils", "_rebuild_parameter"): rebuild_parameter}
class U(pickle.Unpickler):
    def find_class(s, mod, name):
        if (mod, name) in ALLOWED: return ALLOWED[(mod, name)]
        if mod == "torch" and (name.endswith("Storage") or name in ("float32", "bfloat16", "float16", "int64", "bool")): return DT(name)
        raise pickle.UnpicklingError(f"blocked global {mod}.{name}")
    def persistent_load(s, pid): return Storage(pid[1], pid[2], pid[4])
z = zipfile.ZipFile(sys.argv[1]); obj = U(z.open([n for n in z.namelist() if n.endswith("data.pkl")][0])).load()
# walk obj -> print key, shape; sum numel per (member, component)
```

**Output summary.**

Bridge, per member:

| component | params |
|---|---|
| `vision_poolings` | 8,410,624 |
| `text_pooling` | 8,410,624 |
| `trajectory_encoder` | 8,411,136 |
| `input_projection` | 524,800 (512×1024 + 512) |
| `single_step_action_encoder` | 4,096 (512×7 + 512) |
| `text_aware_visual_extraction` | 589,825 (`pos_emb` 576×1024 buffer + temperature) |
| `action_padding_value` | −5.0 |

The Bridge checkpoint has **no positional-embedding key and no logit_scale**, and holds **3 members**.

PolaRiS, per member: `input_projection` 786,944 (512×1536), `single_step_action_encoder` 4,608 (512×8), **`action_pos_embedding` (1, 31, 512)**, `logit_scale` 1, all else identical; **3 members**.

The full key and shape listing is reproducible with the script above. Both sha256 values match the HF LFS oids, as listed in the header.

---

## Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| **this research note** | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/RH_cover_vla.md` | **staged** (`git ls-files --cached` verified), not committed |
| checkpoint inspector script | inlined above (Appendix A); scratch copy at `/tmp/claude-0/streamC/scripts/pt_keys.py` | **in repo via this note**; the scratch copy is disposable |
| paper full text (HF markdown of arXiv v2) | scratch `/tmp/claude-0/streamC/paper/paper.md` | scratch only. **Not banked: it is third-party copyrighted text**, re-fetchable from the URL in the header |
| CoVer code clones | scratch `/tmp/claude-0/streamC/repo/cover-vla` @ `21a9960`, `/tmp/claude-0/streamC/repo/cover-vla-polaris` @ `2cba5bc` | scratch only; re-clonable at the pinned commits |
| CoVer weights (Bridge, PolaRiS) | scratch `/tmp/claude-0/streamC/weights/*.pt` (sha256 in header) | scratch only; never loaded with torch, never executed |
| HiCAP / REF-F / registry files | read only, **unchanged** | — |

**Integration items (escalated, not written as "please merge"):**
- E1: three paper-sentence corrections.
- E2: an Alpamayo re-run decision for any Alpamayo-candidate verifier.
- E3: add the anti-echo rule and the sticky-off control to PREREG_HICAP and NC6.
- **P1–P5 need a PREREG_HICAP amendment** (dated, with a reason) before any of them trains or is scored.
