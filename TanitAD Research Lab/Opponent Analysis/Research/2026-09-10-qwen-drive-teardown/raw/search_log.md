# Search log — Qwen-Drive-1.0-4B teardown, 2026-09-10

`⭐ PI-requested mid-pass. Retrieval by route 5 (byte download + local PyMuPDF extraction), the method promoted by today's Data Engineering package.`

| # | query / route | hits | outcome |
|---|---|---|---|
| Q-1 | `Qwen-Drive-1.0-4B autonomous driving model paper` | 9 | arXiv **2609.00111** (2026-09-02) · HF `Qwen/Qwen-Drive-1.0-4B` · ModelScope · GitHub `QwenLM/Qwen-Drive-1.0` · Apache 2.0 · Qwen Team + Huazhong University of Science and Technology |
| Q-2 | `arxiv.org/pdf/2609.00111` → `urllib` byte download → PyMuPDF | **full text** | **32,477,575 B, `%PDF-` header, valid `%%EOF`, 40 pp., 155,310 chars extracted.** Route 5 first try |
| Q-3 | full-text section index | — | §2.1 architecture · §2.2 training recipe · §2.3 data recipe · §3.1–3.4 experiments and ablations · §5 limitations |
| Q-4 | targeted extraction: Table 5 (PAI-AV open-loop), Table 6 (NAVSIM v1.1 navtest), **Table 7 (AlpaSim closed-loop)**, Table 8 (Stage-2 mixture), Fig. 11 (RL reward ablation), data-scaling series | — | all six extracted verbatim |

## ⛔ Version stamp (rule LI10-1, adopted today)

**`2609.00111v1`, retrieved 2026-09-10.** The version list was **not** checked for a v2/v3 — ⚠️ **declared, because today's Kairos near-miss was caused by exactly this omission.** Every number in the package is stamped v1 and must be re-checked against the current version before it enters `MODEL_REGISTRY.md` or the paper.

## Empty / not found

| id | item | status |
|---|---|---|
| **QD-E1** | a parameter count for the BEV perception head and the Planning Expert separately | ⛔ **NOT STATED.** Only the 5.0 B total, *"excluding the LLM token embeddings"*. The head/expert split is unavailable, so no per-module efficiency comparison against our sub-300 M positioning is admissible |
| **QD-E2** | inference latency or throughput, for any component | ⛔ **ABSENT FROM THE ENTIRE PAPER.** No ms figure, no FPS, no hardware named. ⚠️ **A 5.0 B model with a 10-step flow sampler carries an obvious deployment question that the paper does not address at all** — and note this is the opposite of Drive-HWM, which reports latency but not parameters |
| **QD-E3** | whether the depth net's `N_d` (number of depth bins) is given | ⛔ **NOT STATED**, though the mechanism is fully described |
| **QD-E4** | an ablation isolating the *reasoning* condition's contribution to closed-loop AlpaSim | ⛔ **NOT PRESENT.** Table 7 reports SFT-with-reasoning and RL only — there is **no** SFT-without-reasoning closed-loop row, so the reasoning channel's closed-loop value is unmeasured. Given that reasoning is worth 0.01 m open-loop, this is the missing row that matters most |
| **QD-E5** | AlpaSim fidelity / sim-to-real characterisation (register debt **D-12**) | ⛔ **STILL ABSENT.** They *use* AlpaSim and report six metrics on 916 scenarios, but state no fidelity limitation. D-12 stands — ⭐ though the protocol itself is now documented well enough for us to reproduce |
