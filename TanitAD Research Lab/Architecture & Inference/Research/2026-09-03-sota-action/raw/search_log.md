# Search log — 2026-09-03 SOTA pass, theme 3 (prediction / action sensitivity)

Every query of this pass and every rejected or unbanked source. The web searches were run by one search subagent (transcript `subagents/agent-a3dfa82bd3057d1e6.jsonl`, 2026-09-03 ~01:14 Berlin) for all four themes at once; the strings below are verbatim from that transcript. Reading was done from PDFs banked with `tools/kb_add.py` and text-extracted off-Drive with `pypdf` (the Read tool cannot render PDFs on this box — no poppler).

## Queries (verbatim, in order; themes served)

| # | tool | query | theme(s) |
|---|---|---|---|
| 1 | WebSearch | `JEPA latent world model multi-step rollout drift regularization ablation arXiv 2026` | 1 |
| 2 | WebSearch | `latent world model rollout consistency loss EMA teacher drift on-manifold ablation V-JEPA 2-AC PLDM DINO-WM` | 1 |
| 3 | WebSearch | `frozen DINOv2 versus fine-tuned encoder world model driving ablation action-conditioned latent prediction arXiv 2026` | 2 |
| 4 | WebSearch | `action-conditioning collapse latent world model action dropout inverse dynamics auxiliary counterfactual consistency arXiv 2026` | **3** |
| 5 | WebSearch | `linear probe transition latent world model decodability ego velocity agents free space probe 2026 arXiv self-supervised` | 4 |
| 6 | WebSearch | `LeWorldModel arXiv 2603.19312 SIGReg JEPA world model action sensitivity` | **3**, 1 |
| 7 | WebSearch | `driving world model latent JEPA action-conditioned 2026 arXiv encoder fine-tuning decodable latent action` | 2, **3** |
| 8 | WebSearch | `"world model" latent prediction "action sensitivity" OR "action-conditioned" ablation "inverse dynamics" 2026 arXiv JEPA collapse` | **3** |
| 9 | WebSearch | `driving world model action classifier-free guidance action dropout controllability metric ablation arXiv 2026 latent` | **3** |
| 10 | WebSearch | `latent world model rollout error accumulation noise injection training multi-step consistency ablation table 2026 arXiv JEPA horizon curriculum` | 1 |
| 11 | WebSearch | `Drive-JEPA arXiv 2026 video JEPA trajectory distillation end-to-end driving` | 2 |
| 12 | WebSearch | `world model latent "counterfactual" action sensitivity metric zero-action baseline displacement evaluation arXiv 2026` | **3** |
| 13 | WebSearch | `frozen DINOv3 world model predictor 2026 arXiv latent prediction planning ablation encoder` | 2 |
| 14 | WebSearch | `latent world model probe predictor output versus current frame persistence baseline physical state linear MLP probe trajectory split 2026` | 4 |

Second probes for the absence claims in `RESULT.md` §6 were greps over the 36 extracted full texts (scratchpad `txt/`), not web searches; the grep strings are named there.

## Sources REJECTED as PUBLISHED-SECONDARY (surfaced by the searches; never cited)

Medium posts, emergentmind topic pages, alphaxiv overviews, ResearchGate mirrors, Substack newsletters, quantumzeitgeist, Hugging Face "papers" pages, X/Twitter threads. Rule: a citation not banked as a PDF is inadmissible; these were used at most to find an arXiv id.

## Relay corrections

- The search subagent's summary named lib `2607.05238` "MoP-JEPA (hard-assigned predictor mixtures)". The banked PDF is **"Branch-JEPA: Finite-Support Predictive Distributions for JEPA World Models"** (Argoverse 2, K weighted latent successors). The library note written at banking time carries the wrong name and is corrected in this pass by re-running `kb_add.py --note`; the entry's `tags`/`cited_by` were right.
- `2606.15032` was expected (from the first-half read) to define an "action-sensitivity ratio" in §5; the extracted text contains no such phrase (grep 0 hits). What it does contain and what is cited: the L0–L7 ladder, objective mismatch, interventional do(a), pipeline entanglement, exploitability gap, and "counterfactual" outcome comparisons (9 hits).

## Candidates surfaced but NOT banked (not read, not cited)

`2607.25337`, `2604.02029`, `2603.22281` (ThinkJEPA), `2606.23444` (SkyJEPA), `2606.27014`, `2607.06401`, `2607.04546`, `2603.12864`, `2606.12987`, `2603.09086`, `2409.15730`, `2412.19505`, `2512.10226`, `2604.22748`, `2503.18938`, `2602.06130`, `2605.01694`, `2510.26433`, `2606.20627`, `2607.17973`, `2511.10894`, `2606.30544`, `2605.21800`, `2602.24181`, `2501.08118`, `2506.01600`, `2602.22010`, `2601.22032` (Drive-JEPA — existence confirmed by search #11 only; frozen-vs-trained question left to theme 2's search log).

## Banked but NOT read in this pass

- `2607.09185` (Causally Debiased Latent Action Model): banked for this theme, never extracted or read; its `cited_by` field already carries this package's `RESULT.md` path from the banking step (the tool appends, it cannot remove). Treat as a false citation record until it is read; the library note is updated in this pass to say so.

## Reading mechanics

- PDFs copied off G: to the scratchpad with size checks (G: flaps, Errno 22); text extracted with `pypdf` 6.14.2 with `===== PAGE n/N =====` markers.
- Texts longer than the per-read cap were read in two chunks: `2606.20104`, `2608.06706`, `2606.07687`, `2605.09701`, `2605.25313`, `2606.15032` (theme 3); `2608.24044`, `2607.05238`, `2603.19312` (theme 1); `2512.24497` (theme 2).
- Load-bearing numbers quoted in `RESULT.md` were re-checked by grep against the extracted texts before writing (all present).
