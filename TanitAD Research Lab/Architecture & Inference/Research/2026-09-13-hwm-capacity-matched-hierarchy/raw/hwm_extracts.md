# Verbatim extracts — arXiv 2604.03208v2 (16 Jun 2026), text extracted locally with PyMuPDF from the banked PDF (sha256 a3217f9f…)

Retrieval: `fitz.open(<banked pdf>)`, full 29 pages, 85,810 characters. Table cells below are transcribed from the extracted text stream (one cell per line in the PDF text layer).

## Appendix D.2 "Model Size vs. Hierarchy" (p. 23)

> "To disentangle these effects, we compare hierarchical models against single-level world models with matched or larger parameter counts. As shown in Table 16, increasing the capacity of single-level models does not improve performance—in fact, it often degrades it"

Table 16 (a) Push-T

| Method | Params | H=10 | H=15 |
|---|---|---|---|
| DINO-WM (flat) | 44M | 55% | 17% |
| DINO-WM (flat) | 98M | 35% | 15% |
| DINO-WM (hierarchy) | 94M | 78% | 61% |

Table 16 (b) Diverse Maze

| Method | Params | D∈[9,12] | D∈[13,16] |
|---|---|---|---|
| PLDM (flat) | 54k | 85% | 63% |
| PLDM (flat) | 178k | 82% | 59% |
| PLDM (hierarchy) | 182k | 95% | 83% |

No interval and no seed count are printed with Table 16.

## §3.2 Push-T, Table 2 (p. 7)

| Method | d=25 | d=50 | d=75 |
|---|---|---|---|
| GCIQL | 40% | 25% | 7.5% |
| HIQL | 55% | 30% | 20% |
| HILP | 25% | 13% | 0% |
| DINO-WM | 84% | 55% | 17% |
| DINO-WM (hier.) | 89% | 78% | 61% |

## Frozen encoder (p. 6)

> "The low-level planner uses DINO-WM with a frozen DINOv2 encoder and a 25M-parameter causal ViT world model over short action–latent contexts [62]. The high-level model operates in the same latent space over randomly sampled waypoint sequences: it scales the ViT world model to 75M parameters and uses a transformer-based action encoder to compress primitive action chunks into latent macro-actions of dimension 4."

## Franka reliability (App., Table 17 context)

> "We run N = 5 independent trials per (start, goal) configuration … yielding 50 total trials per object on pick-&-place and 35 on drawer." (Clopper–Pearson 95 % intervals reported there, not for Table 16.)
