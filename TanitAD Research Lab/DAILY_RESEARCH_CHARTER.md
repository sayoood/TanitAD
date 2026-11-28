<title>DAILY RESEARCH CHARTER - what the Research Lab owes the programme every single day</title>

# DAILY RESEARCH CHARTER

`TanitAD Research Lab · BINDING. Issued by the PI (Sayed), 2026-08-31. Recorded by the Master Mind.`

> **This charter SUPERSEDES the research-scope half of the daily Lab trigger.** The trigger's
> mechanics (dedup gate, staging, backlog motions, deliverable manifest) stand unchanged. What
> changes is the SCOPE and the SHAPE of the research, because the delivered scope was too narrow.

---

## 0. The PI's statement, verbatim (2026-08-31)

> *"A detailed paper, publications, blogs, news, scientific work which is relevant for all our
> works, streams, flywheels in highest quality and also very broad. I need it in high structured
> reports which grow incrementally, extend our knowledge base, the downloaded raw documents must
> be saved, etc... An analysis regarding relevance, most important consequences, combinations with
> our thoughts, chances and risks and possible experiments for the lab and the wheels. Its
> explicitly wished to identify the ideas from neighbouring disciplines like vlm, llm, post
> transformer architecture, efficient decoding, training, post training, RL, finetuning,....and
> transfer it to tanitAD."*

## 1. ⛔ The four defects this charter exists to fix (MEASURED on the 2026-08-31 pass)

| # | defect | measurement |
|---|---|---|
| D1 | **Literature was optional per package.** | 2 of 4 packages banked any primary; Deployment banked 0 (a microbenchmark), Data Eng banked 0 (a corpus-descriptor re-read). |
| D2 | **No news / blog / release-note channel at all.** | 1 non-paper source across the whole day (the NAVSIM README changelog). Zero lab announcements, zero release notes, zero engineering blogs. |
| D3 | **Zero neighbouring-discipline coverage.** | Every 2026-08-31 primary is a driving or world-model paper. No VLM, LLM, post-transformer, decoding, post-training, RL, diffusion, tokenizer or PINO item exists in the day's output. |
| D4 | **Reports reset daily instead of growing; empty searches unnamed.** | `raw/search_log.md` absent from all four packages; `grep -ic "empty\|no results\|not found"` = 0 and 1. An unrecorded empty search is a false-absence generator — `CLAUDE.md` rule 2. |

⚠️ **D3 is the expensive one.** The programme's own history says so: the levers that actually moved
TanitAD came from OUTSIDE driving — the speed-as-action-channel fix, the EMA-teacher question,
VICReg placement, jumpy/temporally-abstract models. **A sweep that only reads driving papers is
structurally unable to find the next one.**

---

## 2. The track taxonomy — 22 tracks, three bands

⭐ **Every track is SCANNED every day. Tracks are DEEP-READ on rotation.** SCAN = query, triage
titles/abstracts, record hits *and misses* in the search log. DEEP = read the primary, bank it with
`kb_add.py`, write the five-dimension analysis of §4.

### Band A — core (DEEP every day, all five)

| # | track | why it is daily |
|---|---|---|
| A1 | World models + world-action models (driving and general) | the programme IS one |
| A2 | JEPA family, joint-embedding + predictive architectures | our objective family |
| A3 | Vision encoders / visual representation learning | our trunk, and our measured ceiling |
| A4 | Vision-action + vision-language-action models (VLA) | the action-channel question is live |
| A5 | Benchmarks + evaluation, driving and embodied | G3 has no lane without it |

### Band B — neighbouring disciplines (SCAN daily, DEEP ≥3 per day on rotation)

⭐ **This band is the PI's explicit mandate. A day that deep-reads zero Band-B tracks is a FAILED
pass, regardless of how good Band A was.**

| # | track | the transfer question to ask of every hit |
|---|---|---|
| B1 | VLM / multimodal / omni models | does it give us a semantic channel the trunk lacks? |
| B2 | LLM architecture + post-transformer (SSM, linear/sparse attention, hybrids, recurrent depth) | does it change what a ~300 M-param sequence model can hold? |
| B3 | Efficient decoding + inference (speculative, KV, sparsity, quantisation) | does it buy rollout depth inside the 100 ms budget? |
| B4 | Efficient training (optimizers, muP, parallelism, low precision, curricula) | does it buy arms per GPU-day on a fixed fleet? |
| B5 | Post-training, RL, finetuning (RLVR/GRPO/DPO, distillation, PEFT) | can a trained WM be improved without a retrain? |
| B6 | Self-improving systems (self-play, self-distillation, generated curricula) | can the Lab close its own loop? |
| B7 | Diffusion models + flow matching | multi-modal futures without the blurry mean |
| B8 | Tokenizers + discrete representations (VQ/FSQ, video tokenizers, byte/patch) | our latent geometry is a tokenizer question |
| B9 | Data curation + dataset design | our corpus is fixed by parity; curation is the free lever |
| B10 | Semantic search / retrieval / embeddings | retrieval-augmented driving memory |
| B11 | Physics-informed neural operators + scientific ML | dynamics priors the predictor now learns from scratch |
| B12 | Memory, long context, state | 6 s tactical vs 30 s strategic horizon |
| B13 | 3D, geometry, occupancy, rendering (gsplat/NeRF) | BEV localisation — our measured geometry ceiling |

### Band C — non-paper sources (SCAN every day, every track)

| # | track | sources |
|---|---|---|
| C1 | Frontier-lab + AV-company releases and announcements | model cards, launch posts |
| C2 | Engineering blogs + release notes | NVIDIA/TensorRT/JetPack, PyTorch, HuggingFace, vLLM, benchmark repos |
| C3 | Regulatory + safety (UNECE, NHTSA, EU AI Act where it touches AV) | standards bodies |
| C4 | Community signals | leaderboards, challenge results, dataset releases |

⚠️ **Band C findings are NOT downgraded for being non-papers** — but they carry
`PUBLISHED-BLOG` / `PUBLISHED-RELEASE-NOTE` as their evidence class, never bare `PUBLISHED`, and a
blog claim may never decide a GPU-day on its own.

---

## 3. The incrementally-growing artifact (the PI's "reports which grow")

```
TanitAD Research Lab/Frontier Scan/
  TRACKS.md                  <- the 22-track coverage ledger: last SCAN, last DEEP, staleness
  LEDGER_<track>.md          <- ONE per track. APPEND-ONLY. The running report for that track.
  Daily/<YYYY-MM-DD>/
    RESULT.md                <- the day's findings-first report
    raw/search_log.md        <- EVERY query, its hit count, and every EMPTY search named
    raw/*.md                 <- verbatim quotes with retrieval method
```

⛔ **`LEDGER_<track>.md` is APPEND-ONLY and is the deliverable that GROWS.** A daily `RESULT.md` is
a snapshot; the ledger is the accumulating state of the art in that track, **with our position in
it**. Never rewrite a ledger's history — correct it with a dated correction entry, the
`RETRACTION_LOG` discipline.

⛔ **Every deep-read primary is BANKED**: `python tools/kb_add.py <id-or-url> --tag <track> --note
"<one-line finding>" --cited-by "<RESULT.md repo path>"`. Cited-but-unbanked = incomplete. Band-C
sources that are not PDFs are banked with `--local` after saving the page, or recorded in the
search log with their retrieval date if unbankable — and then they are `RELAYED`, not `PUBLISHED`.

---

## 4. The five analysis dimensions — MANDATORY per deep-read item

**A banked paper with no analysis is a download, not research.** Every deep-read item carries:

| # | dimension | the question |
|---|---|---|
| 1 | **RELEVANCE** | which stream / FlyWheel / gate item does this touch, and how directly? Rate `⭐⭐⭐` direct lever · `⭐⭐` informs a live decision · `⭐` context. |
| 2 | **CONSEQUENCE** | what does it change for TanitAD if true? Name the doc, gate row, or register claim it moves. |
| 3 | **COMBINATION** | how does it combine with OUR thoughts — our measured results, our open hypotheses, our other findings? **This is where transfer happens; a Band-B item with no combination line has not been transferred.** |
| 4 | **CHANCES / RISKS** | the upside if it holds; the failure mode, the cost, and what would make it NOT transfer to us. Both halves, always. |
| 5 | **EXPERIMENT** | a concrete, cheap, pre-registerable test — for the Lab (0-GPU or 4060) or for a named FlyWheel, **with its discriminating outcome committed in advance**. |

⭐ **Cap: at most 3 recommendations per package** (unchanged) — but the EXPERIMENT column is *per
item*, because a proposed experiment is not a recommendation until ranked. Experiments go to
`LAB_BACKLOG.md` under `## PROPOSED (unranked)`, never self-ranked.

---

## 5. Quality bar — inherited, restated because breadth tempts shortcuts

1. **Evidence class on every number**: MEASURED / PUBLISHED / PUBLISHED-BLOG / PUBLISHED-RELEASE-NOTE / RELAYED / INHERITED / ESTIMATED / HYPOTHESIS.
2. **Abstract-only is declared.** `PUBLISHED lib <key> abstract-only` — banked, but not read. It may not decide a GPU-day.
3. ⛔ **RELAYED never upholds or retracts a novelty claim.** (The 2026-08-31 Opponent package got this right; it is now the rule.)
4. **Every empty search is named** in `raw/search_log.md` with its exact query. Absence found at one location is not absence — two probes minimum before writing "X does not exist".
5. **Breadth may not buy shallowness.** Better 5 deep-read items with all five dimensions than 30 titles with none. **The SCAN band carries the breadth; the DEEP band carries the depth.**
6. **No spend, no Thor, no pod.** Dev-box RTX 4060 only, and only when `nvidia-smi` shows no python compute.

---

## 6. Daily shape

| # | step | output |
|---|---|---|
| 1 | dedup gate (has today already run?) | stop, or proceed |
| 2 | read `LAB_ASKS.md` OPEN rows, `LAB_BACKLOG.md` INJECTED lane, `TRACKS.md` staleness | the day's DEEP list |
| 3 | **SCAN all 22 tracks** | `raw/search_log.md` — every query, hit count, and every empty named |
| 4 | **DEEP-read**: Band A ×5 + Band B ×≥3 (rotation, staleness-ordered) + Band C sweep | banked primaries + five-dimension analysis |
| 5 | **APPEND to each touched `LEDGER_<track>.md`** | the growing report |
| 6 | domain work packages (unchanged: SPEC/PLAN/RESULT/COMMS + raw) | as before |
| 7 | KNOWLEDGE_BASE one-liner per package; backlog motions; `TRACKS.md` update | same turn |
| 8 | `LAB-RUN-<NNN>.md` summary + stage everything | manifest with `git ls-files --cached` |

⛔ **A pass that produces no Band-B deep-read, or no Band-C sweep, or no ledger append, is
INCOMPLETE and says so in its own summary.** Failing loudly is the contract; quietly narrowing
scope is what produced defects D1–D4.
