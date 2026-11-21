# Search log — 2026-09-03 SOTA pass, theme 4 (representation and decodability)

Every query of this pass and every rejected or unbanked source, with the reason. **Two attempts:** the first (2026-09-03 ~00:58–04:05 Berlin) was killed by the model limit after banking six primaries and appending three `KNOWLEDGE_BASE.md` entries (now in history at commit `4cd3cdb`); it left no `RESULT.md` or `PROPOSED_HYPOTHESES.md`. The second attempt (this one, from ~10:10 Berlin) inherits that banking, re-reads what it quotes, and adds the queries below.

Reading was done from banked PDFs, text-extracted **off-Drive** with `pypdf` into the session scratchpad (`.../scratchpad/txt/`), per the G:-flap rule. `python tools/kb_add.py --verify` after banking: **verified 334 entries, 0 orphan(s), 0 problem(s)** — content, not presence.

## Attempt 1 — inherited queries (run by the shared search subagent for all four themes, 2026-09-03 ~01:14; strings verbatim from theme 1's log)

| # | tool | query | theme(s) |
|---|---|---|---|
| 5 | WebSearch | `linear probe transition latent world model decodability ego velocity agents free space probe 2026 arXiv self-supervised` | **4** |
| 14 | WebSearch | `latent world model probe predictor output versus current frame persistence baseline physical state linear MLP probe trajectory split 2026` | **4** |

Banked by attempt 1 under `sota-2026-09-03-decodability`: `2504.03861`, `2510.26782`, `2603.20327`, `2603.21546`, `2603.24581`, `2608.05720`.

## Attempt 2 — new queries (verbatim)

| # | tool | query | question answered |
|---|---|---|---|
| 15 | WebSearch | `latent world model linear probe protocol leave-one-episode-out grouped cross-validation pseudo-replication clustered data arXiv 2026` | (a) |
| 16 | WebSearch | `probe passes on privileged oracle input ablation world model representation "input ablation" deployment inputs unavailable at test time shortcut probing 2026 arXiv` | (c) |
| 17 | WebSearch | `decoding free space occupancy lane geometry from frozen versus shaped latent world model probe BEV segmentation mIoU numbers 2026 arXiv driving` | (e) |
| 18 | WebSearch | `transition-level probing latent difference decoding Delta-JEPA successor world model probe delta z displacement readout 2026 arXiv` | (b) |
| 19 | WebSearch | `representation quality metric RankMe effective rank dataset dependence corpus matched reference comparison across datasets self-supervised evaluation 2026` | (d) |
| 20 | WebSearch | `"ego status" ablation open-loop end-to-end driving planner shortcut ego state removal AUC collapse 2026 arXiv navigation command ablation` | (c), our domain |
| 21 | WebSearch | `world model latent probe "control task" selectivity random-label baseline probe capacity Hewitt Liang designing interpreting probes representation` | (a) probe capacity |
| 22 | WebSearch | `"Intervention Gap in Latent World Models" 2608.29998 grouped nested cross-validation ridge probe protocol` | (a) targeted follow-up on #15's top hit |
| 23 | WebSearch | `probing world model latent representations "leakage" instruction goal shortcut probe control condition 2026 arXiv compact world model spatial relations` | (c) targeted follow-up on #16 |
| 24 | WebSearch | `"effective rank" OR "participation ratio" representation collapse metric depends on dataset diversity number of episodes not comparable across corpora 2026 arXiv world model` | (d) |

## Banked in attempt 2 (five new; three re-tagged)

| key | why banked | read |
|---|---|---|
| `2608.29998` The Intervention Gap in Latent World Models | the only primary that states the uncertainty-unit rule as a protocol AND publishes a capture→resolvability→propagation gate order | full |
| `2607.06925` Grounding Spatial Relations: Instruction Leakage and a Goal-Free Dynamics Fix | the direct published analogue of `D-REFAV1-TAC-DECODER-PANEL`, with the two controls we lack | full |
| `2608.10145` The Evaluation Protocol Determines the Result | protocol-dependence of probe values, and effective rank at fixed decodability | full |
| `2605.06388` Reconstruction or Semantics? | the frozen-readout construction (train on real latents, apply unchanged to predicted latents) with encoder→WM numbers | full (chunks) |
| `2608.06799` Is Forward Prediction Enough? (PSG-JEPA) | the second independent transition-probe protocol, on held-out episodes, with a multi-horizon net-state-change target | full (chunks) |
| `2312.03031` Is Ego Status All You Need? | already banked; **re-tagged** `sota-2026-09-03-decodability` — the deployment-input ablation in our own domain | full (results) |
| `2606.31232`, `2607.27017`, `2210.02885` | already banked; **re-tagged** and read for this theme. `2607.27017` was `INHERITED (abstract-only)` in `SPEC.md` §1 — **that row is discharged: read in full here** | as stated in `RESULT.md` §7 |

## Sources REJECTED as PUBLISHED-SECONDARY (used at most to locate an arXiv id; never cited)

`emergentmind.com` (RankMe topic page, surfaced by #19), `liner.com` (RankMe "quick review", #19), `researchgate.net` (RankMe PDF mirror, #19), `scribd.com` (a mirror of `2312.03031v2`, #20), `alphaxiv.org` (`2606.31232`, `2312.03031`), `dl.acm.org` / `proceedings.mlr.press` (RankMe proceedings pages — the arXiv PDF was banked instead), `github.com/LMD0311/Awesome-World-Model` and `github.com/vasgaowei/BEV-Perception` (link lists, #17), `openreview.net` PDF for "Co-evolving latent action world models" (#22 — anonymous, no arXiv id resolved).

## Candidates surfaced but NOT banked, with the reason

| candidate | surfaced by | reason not banked |
|---|---|---|
| `2605.31111` Subspace-Decomposed JEPAs | #18 | disentangles progression from content; no transition-level probe table in the returned material — **not read, not cited**. Open candidate for a later pass |
| `2605.15725` DiLA: Disentangled Latent Action World Models | #15 | latent-action factorisation, not a probe-protocol paper |
| `2602.10104` Olaf-World | #15 | latent-action orientation; no probe protocol |
| `2210.04482` leave-group-out CV for latent Gaussian models | #15 | statistics methodology outside ML representation learning; the grouped-CV point is already carried by `2608.29998` with a world-model instantiation. **This is the one rejection that costs something** — a dedicated grouped-CV primary would strengthen F1 |
| PMC8671136 (leave-one-site-out CV in resting-state fMRI) | #15 | neuroimaging; correct on the statistics but not a world-model primary, and citing it would import a different field's conventions without a bridge |
| `2508.02159` PIGDreamer (privileged information guided WM) | #16 | privileged info as a TRAINING aid for safe RL; it does not run a deployment-input ablation on a probe |
| `2606.31422` Budgeted Environment Probing for World-Model Calibration | #16 | "probing" means environment interaction budgeting, not representation probing — a homonym |
| `2606.22966` Oracle-Level Integrity Attacks on Imagine-then-Act World Models | #16 | adversarial security framing; "oracle" is the attacker's capability, not a model input |
| `2509.21344` Linear probes rely on textual evidence (leakage mitigation in LMs) | #23 | language-model domain; the mechanism (probe reads the surface text) is the same family as `2607.06925`'s transcription, but importing an LM result into a driving claim needs a bridge this pass does not have. **Named as a boundary, not cited** |
| `2604.02608` Steerable but Not Decodable: Function Vectors Beyond the Logit Lens | #21 | LM interpretability; relevant to "decodable ≠ used" but not a world model |
| `2003.12298` Information-Theoretic Probing with MDL; `2009.07364`; `2207.01736` Probing via Prompting | #21 | the probing-methodology canon, all NLP. `2607.27017` already imports its central caution (*"probe results reflect probe capacity as much as representation content"*, citing Hewitt & Liang 2019 and Belinkov 2022) into the world-model setting, which is the bridged form. **Not banked; if the paper's methods section needs the canonical citation, bank `1909.03368` (Hewitt & Liang) then** |
| `2605.08732` Latent Geometry Beyond Search; `2605.06388`'s neighbours `2603.12655` VGGT-World, `2606.24353`, `2604.28196` HERMES++, `2311.16038` OccWorld, `2505.05512`, `2601.01577` HanoiWorld | #17 | occupancy/BEV generation and geometry world models; none reports a **frozen-vs-shaped latent** decodability comparison with numbers, which is what (e) asked for. `2605.06388` was banked instead because it is the controlled encoder-swap study |
| `2605.01694` Latent State Design under Sufficiency Constraints; `2606.04130` CLAW; `2606.00113` WM survey | #22 | surfaced by a targeted query for a different paper; not read |
| `2604.03809`, `2605.30524`, `2605.23191`, `2510.17269` FineVision, `2401.10474` LDReg, `2604.16027`, `2411.04832` | #24 | effective-rank / collapse work in LLM, recommender and RL-plasticity settings. FineVision's dataset-tier observation is the closest analogue to H-RANK-23 but is about **training data diversity**, not about a representation-quality gate; citing it would be a scope error of the family `CLAUDE.md` warns about |

## Second probes (greps over the extracted texts; scratchpad `txt/`, 40 files)

| absence claim | probe 2 | result |
|---|---|---|
| clip/episode-disjoint splitting is rare | `grep -c -iE "split (train/test)? ?by (trajectory\|episode)\|episode-split\|held-out (test )?episodes\|disjoint episode\|leave-one-(episode\|trajectory\|clip)-out\|grouped (nested )?cross-validation\|group k-?fold"` | **5 files**: `2607.05238`, `2607.27017`, `2608.06799`, `2608.24044`, `2608.29998` |
| only one primary declares the uncertainty unit | `grep -aiE "uncertainty unit\|pseudo-?replicat\|independent replication\|whole-family bootstrap"` | **2 files**: `2608.29998` (`:210-212` states the rule; `:362`, `:476`, `:519`, `:614`, `:714`, `:1090`, `:1102`, `:1115` apply it) and `2603.20327` (`:1102`, as a self-declared limitation) |
| the deployment-input ablation is reported | `grep -c -iE "withheld\|counterfactual (goal\|instruction)\|blank image\|input ablation\|passthrough\|untrained (copy\|encoder)\|random projection"` | **8 files**, concentrated in `2607.06925` (18) and `2607.27017` (13); `2312.03031` (3) is the driving-domain instance |
| no matched-reference convention for rank | `grep -a -c -iE "effective rank\|participation ratio\|RankMe\|spectral entropy"` | **6 files**: `2210.02885` (199), `2605.09241` (6), `2608.10145` (6), `2608.24044` (6), `2605.25313` (5), `2607.05238` (2). Read: none defines a cross-corpus reference; `2210.02885` restricts to within-method comparison |
| the endpoint-shuffled control for a difference target | full reads + the passthrough/persistence greps above | **0 primaries** — the nearest constructions are `2607.27017`'s untrained-passthrough bound and `2608.29998`'s zero-effect/persistence controls, neither of which permutes the endpoint. Recorded as **TanitAD-novel** |

## Numbers re-checked by grep against the extracted texts before they entered `RESULT.md`

`0.90 → 0.27` and `94.5 / 2.3` (`2607.06925` abstract, §5.1, Table 1); `0.04 / 0.58 / 0.89 / 0.98` and `0.10 vs 0.19 (static 0.21)` (`2607.27017` Fig. 1, §4.2); `VFX 0.47/0.51 vs persistence 0.24/0.31` and the paired CIs `[+0.01,+0.17] / [+0.14,+0.28] / [−0.55,−0.15]` (`2607.27017` Table 15, `:1070-1085`); `0.94 / 0.97` untrained passthrough (`:463-466`); `18.6 → 67.8`, `16.5 / 11.9`, `0.9977 / 0.9971 / 0.9994`, `0.9290 / 0.8982` (`2608.10145` Table 2, §4.1, §5.4); `0.355/0.813 … 0.016/0.992` (`2606.31232` Table 5) and `0.078/0.960 … 0.004/0.998` (Table 4); `0.44/0.47`, `0.20/0.12`, `0.69/0.76` (`2608.06799` Table 2); `0.021/0.102` vs `0.507/0.626` (`2608.29434`, verified in this pass); `62.5 % → 5 %` and `p = 0.357` (`2603.20327` `:552-555`, `:573-575`); `L2 0.37 → 0.46` with `NDS 45.5 → 0.0`, `L2 0.37 → 6.16`, Ego-MLP `0.35` (`2312.03031` Tables 1–2); the RankMe scoping sentence (`2210.02885` `:294-297`); `capture 0.579-0.609 / real 0.127-0.198 / predicted −6.33..−3.36` (`2608.29998` Table 1).

## Reading mechanics

PDFs copied off-Drive with a 6-attempt retry loop (G: flap guard), extracted with `pypdf`. Chunked reads for `2608.29998`, `2608.10145`, `2607.27017`, `2605.06388`, `2210.02885`. `grep` was run with `-a` throughout after `CLAUDE.md`'s own file (and two extracted texts) were reported as *binary* by an un-flagged `grep` — the same under-reporting family as the G:-Drive rule; every absence claim above therefore carries two differently-bound probes (a full read and a flagged grep).
