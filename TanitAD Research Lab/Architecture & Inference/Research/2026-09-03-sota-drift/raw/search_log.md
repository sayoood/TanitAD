# Search log — 2026-09-03 SOTA pass, theme 1 (latent drift over the rollout)

Every query of this pass and every rejected or unbanked source. One search subagent (transcript `subagents/agent-a3dfa82bd3057d1e6.jsonl`, 2026-09-03 ~01:14 Berlin) ran the searches for all four themes; the strings are verbatim. Reading was done from banked PDFs text-extracted off-Drive with `pypdf`.

## Queries (verbatim; theme-1 queries in bold)

| # | tool | query | theme(s) |
|---|---|---|---|
| 1 | WebSearch | **`JEPA latent world model multi-step rollout drift regularization ablation arXiv 2026`** | **1** |
| 2 | WebSearch | **`latent world model rollout consistency loss EMA teacher drift on-manifold ablation V-JEPA 2-AC PLDM DINO-WM`** | **1** |
| 3 | WebSearch | `frozen DINOv2 versus fine-tuned encoder world model driving ablation action-conditioned latent prediction arXiv 2026` | 2 |
| 4 | WebSearch | `action-conditioning collapse latent world model action dropout inverse dynamics auxiliary counterfactual consistency arXiv 2026` | 3 |
| 5 | WebSearch | `linear probe transition latent world model decodability ego velocity agents free space probe 2026 arXiv self-supervised` | 4 |
| 6 | WebSearch | `LeWorldModel arXiv 2603.19312 SIGReg JEPA world model action sensitivity` | 3, **1** |
| 7 | WebSearch | `driving world model latent JEPA action-conditioned 2026 arXiv encoder fine-tuning decodable latent action` | 2, 3 |
| 8 | WebSearch | `"world model" latent prediction "action sensitivity" OR "action-conditioned" ablation "inverse dynamics" 2026 arXiv JEPA collapse` | 3 |
| 9 | WebSearch | `driving world model action classifier-free guidance action dropout controllability metric ablation arXiv 2026 latent` | 3 |
| 10 | WebSearch | **`latent world model rollout error accumulation noise injection training multi-step consistency ablation table 2026 arXiv JEPA horizon curriculum`** | **1** |
| 11 | WebSearch | `Drive-JEPA arXiv 2026 video JEPA trajectory distillation end-to-end driving` | 2 |
| 12 | WebSearch | `world model latent "counterfactual" action sensitivity metric zero-action baseline displacement evaluation arXiv 2026` | 3 |
| 13 | WebSearch | `frozen DINOv3 world model predictor 2026 arXiv latent prediction planning ablation encoder` | 2 |
| 14 | WebSearch | `latent world model probe predictor output versus current frame persistence baseline physical state linear MLP probe trajectory split 2026` | 4 |

## What the theme-1 queries returned and what was done with it

- Banked and read in full: `2608.24044` JEPA-x, `2607.16314` Depth-Reg JEPA, `2607.05238` Branch-JEPA (the relay called it "MoP-JEPA" — corrected), `2608.29029` Flow-JEPA. Already banked and re-read: `2605.09241` Sub-JEPA, `2603.19312` LeWM, `2512.24497` What-Drives-Success, `2506.09985` V-JEPA 2 (§3.1), `2606.26217` Fast-LeWM (banked under the encoder tag), `2504.03861`, `2510.26782` (decodability tag).
- Noise-injection / horizon-curriculum results: query #10 surfaced only the Looped-WM (`2606.18208`) and InfinityDrive (`2412.01522`) lines already banked and quoted in `2026-09-01-longhorizon-bptt-stability` — not re-read; no NEW primary with a curriculum-vs-drift table was found (absence claim #3 in `RESULT.md` §5).
- `2606.31672` WorldRoamBench (banked in an earlier pass; 64 "drift" mentions) is a video-generation benchmark (pixel-space stability), not a latent world model — not cited.

## Sources REJECTED as PUBLISHED-SECONDARY (never cited)

Medium, emergentmind, alphaxiv, ResearchGate, Substack, quantumzeitgeist, Hugging Face "papers" pages, X/Twitter threads; used at most to locate an arXiv id.

## Candidates surfaced but NOT banked (not read, not cited)

`2607.25337`, `2604.02029`, `2603.22281` (ThinkJEPA), `2606.23444` (SkyJEPA), `2606.27014`, `2607.06401`, `2607.04546`, `2603.12864`, `2606.12987`, `2603.09086`, `2409.15730`, `2412.19505`, `2512.10226`, `2604.22748`, `2503.18938`, `2602.06130`, `2605.01694`, `2510.26433`, `2606.20627`, `2607.17973`, `2511.10894`, `2606.30544`, `2605.21800`, `2602.24181`, `2501.08118`, `2506.01600`, `2602.22010`, `2601.22032`.

## Second probes (greps over the 36 extracted texts; scratchpad `txt/`)

- "drift" mentioned in 20 texts; "persistence" in 5 (`2510.26782`, `2606.31672`, `2607.27017`, `2608.06706`, `2608.24044`).
- "predictab*" on the same line as increment / delta / Δz / displacement: 0 hits.
- "EMA" and "drift" in the same text: 6 (`2603.20327`, `2605.25313`, `2606.21775`, `2607.05238`, `2607.27017`, `2608.06706`); none contains an EMA on/off drift row (checked by reading).
- `2506.09985`: "T = 2" and "one recurrent step" present (1 hit each).

## Reading mechanics

As in theme 3's log; chunked reads for `2608.24044`, `2607.05238`, `2603.19312`, `2512.24497`. All load-bearing numbers in `RESULT.md` were re-checked by grep against the extracted texts before writing.
