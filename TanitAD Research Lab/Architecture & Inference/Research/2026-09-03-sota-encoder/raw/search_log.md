# Search log — 2026-09-03 SOTA pass, theme 2 (encoder quality: frozen vs fine-tuned vs shaped)

Every query of this pass and every rejected or unbanked source. One search subagent (transcript `subagents/agent-a3dfa82bd3057d1e6.jsonl`, 2026-09-03 ~01:14 Berlin) ran the searches for all four themes; strings verbatim. Reading from banked PDFs, text-extracted off-Drive with `pypdf`.

## Queries (verbatim; theme-2 queries in bold)

| # | tool | query | theme(s) |
|---|---|---|---|
| 1 | WebSearch | `JEPA latent world model multi-step rollout drift regularization ablation arXiv 2026` | 1 |
| 2 | WebSearch | `latent world model rollout consistency loss EMA teacher drift on-manifold ablation V-JEPA 2-AC PLDM DINO-WM` | 1 |
| 3 | WebSearch | **`frozen DINOv2 versus fine-tuned encoder world model driving ablation action-conditioned latent prediction arXiv 2026`** | **2** |
| 4 | WebSearch | `action-conditioning collapse latent world model action dropout inverse dynamics auxiliary counterfactual consistency arXiv 2026` | 3 |
| 5 | WebSearch | `linear probe transition latent world model decodability ego velocity agents free space probe 2026 arXiv self-supervised` | 4 |
| 6 | WebSearch | `LeWorldModel arXiv 2603.19312 SIGReg JEPA world model action sensitivity` | 3, 1 |
| 7 | WebSearch | **`driving world model latent JEPA action-conditioned 2026 arXiv encoder fine-tuning decodable latent action`** | **2**, 3 |
| 8 | WebSearch | `"world model" latent prediction "action sensitivity" OR "action-conditioned" ablation "inverse dynamics" 2026 arXiv JEPA collapse` | 3 |
| 9 | WebSearch | `driving world model action classifier-free guidance action dropout controllability metric ablation arXiv 2026 latent` | 3 |
| 10 | WebSearch | `latent world model rollout error accumulation noise injection training multi-step consistency ablation table 2026 arXiv JEPA horizon curriculum` | 1 |
| 11 | WebSearch | **`Drive-JEPA arXiv 2026 video JEPA trajectory distillation end-to-end driving`** | **2** |
| 12 | WebSearch | `world model latent "counterfactual" action sensitivity metric zero-action baseline displacement evaluation arXiv 2026` | 3 |
| 13 | WebSearch | **`frozen DINOv3 world model predictor 2026 arXiv latent prediction planning ablation encoder`** | **2** |
| 14 | WebSearch | `latent world model probe predictor output versus current frame persistence baseline physical state linear MLP probe trajectory split 2026` | 4 |

## What the theme-2 queries returned and what was done with it

- Banked in this pass and read in full: `2602.12218` Observer Effect, `2608.29434` point-cloud JEPA WMs, `2507.19468` DINO-world, `2602.18639` invariant/bisimulation representations, `2512.24497` What Drives Success, `2606.26217` Fast-LeWM.
- Banked in EARLIER passes (the Master Mind's 2026-08-31 / 09-01 searches, visible in the main session transcript as queries on FROST-Drive, DINO-WM, V-JEPA 2-AC, Latent-WAM) and RE-READ here from the PDFs: `2601.03460` FROST-Drive, `2411.04983` DINO-WM, `2506.09985` V-JEPA 2 (§3.1 only), `2603.24581` Latent-WAM, `2607.16314` Depth-Reg JEPA (theme 1), `2603.19312` LeWM.
- Search #11 (Drive-JEPA `2601.22032`): existence confirmed; NOT banked, NOT read — its frozen-vs-trained question stays open (RESULT.md §5.3).
- Search #13 returned no primary with a frozen-DINOv3 world model beyond What-Drives-Success (DINOv3 on DROID / Robocasa).

## Sources REJECTED as PUBLISHED-SECONDARY (never cited)

Medium, emergentmind (incl. its DINO-WM topic page), alphaxiv, ResearchGate, Substack, quantumzeitgeist, Hugging Face "papers" pages, X/Twitter threads.

## Candidates surfaced but NOT banked (not read, not cited)

`2601.22032` (Drive-JEPA), `2607.25337`, `2604.02029`, `2603.22281`, `2606.23444`, `2606.27014`, `2607.06401`, `2607.04546`, `2603.12864`, `2606.12987`, `2603.09086`, `2409.15730`, `2412.19505`, `2512.10226`, `2604.22748`, `2503.18938`, `2602.06130`, `2605.01694`, `2510.26433`, `2606.20627`, `2607.17973`, `2511.10894`, `2606.30544`, `2605.21800`, `2602.24181`, `2501.08118`, `2506.01600`, `2602.22010`.

## Verification notes

- Every number in `RESULT.md` §1 was re-checked by grep against the extracted texts before writing; the table rows for FROST-Drive (7.39 / 7.79 / 8.13–1.47 / 8.17–1.04 / 8.24 / width 256 → 7.68), Observer Effect (0.94 → −0.03; 140 % → 18 %; B5–B10), DINO-world Table 1 (Copy-Last / V-JEPA / DINO-world rows), the bisimulation encoder table (No Encoder 0.68 0.44 0.70 0.26 0.36 0.64; DINOv2 0.78 0.8 0.76 0.86 0.78 0.82; SimDINOv2 0.40…; iBOT 0.72…), Utonia-WM / Vox-WM (94.0 / 71.2 / 46.6 / 71.2 vs 81.4 / 54.4 / 35.2 / 63.4) and Latent-WAM (Small 86.3, Small-LoRA 84.7, Base-LoRA 68.5; 88.3 / 88.0 / 89.3) were read from the text lines directly.
- LeWM Fig. 6 bar values are NOT text-verifiable and are stamped INHERITED in `RESULT.md`.
- The Observer Effect's "0.94 → −0.03" is the paper's own caption sentence ("dropping correlation from ρ = 0.94 → −0.03").
