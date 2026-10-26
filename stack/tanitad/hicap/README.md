# tanitad.hicap — reference implementation of the HiCAP architecture (paper `Paper/HiCAP/hicap.tex`)

Status: **architecture, cached-feature dataset, 3-stage training loop and four-family evaluation implemented and unit-tested on CPU with synthetic data (53 tests); nothing trained on real data, backbone wrapper and pod cache builder not run (no GPU).**
Test: `PYTHONPATH=stack:taniteval python -m pytest stack/tests/test_hicap_architecture.py stack/tests/test_hicap_data_train.py stack/tests/test_hicap_full.py`

| module | paper | content |
|---|---|---|
| `vocab.py` | §action tree | v9 classes (3 lat × 7 lon = 21 cells, 19 populated), path/profile extraction, arc-length composition, farthest-point factored codebooks with a standstill sub-vocabulary |
| `model.py` | §backbone | motion adapter, cached-token projector (+age Fourier), nullable ego (v, a_lon, yaw rate — **no curvature**)/nav/route tokens, fusion (d 512, 6 L), level queries, gated audio (zero-init, bit-exact null), Matryoshka readouts, node towers, cost/refine/pair heads, class-routed MoE re-ranker; **56.4 M params** |
| `losses.py` | §scoring and training | level-wide soft-target critics, ancestor sums, masked/unmasked InfoNCE (H-HC4 ablation), consistency KL, goal loss, 3-stage curriculum, lr rule |
| `prune.py` | §pruning | R1 physical masks, exact pair feasibility, guard set, R3 route mask, R5 hysteresis, R6 class-conditional conformal + episode-blocked coverage |
| `decode.py` | §cascade | S0–S4 cascade and decision rule (batch 1 reference) |
| `backbone.py` | §backbone choice | Cosmos3-Edge tap wrapper (blocks 28/42/56, 160-token assertion, min_pixels fix) — **untested** |

| `data.py` | §training-data | v9 reader, per-clip cache contract (rows == episode rows == v9 rows), window dataset (ego = v, a_lon, yaw rate; noised RC-A50 with dropout), factor distances, soft targets, priors, synthetic clips |
| `train.py` | §curriculum | 3 stages, 40 % replay, hard-negative weighting, re-ranker only in stage 3, AdamW + lr rule + one-cycle, checkpoints, `step_s` averaged over `log_every` |
| `evaluate.py` | protocol | longitudinal / lateral / trajectory families with episode-cluster bootstrap (taniteval.ci) and paired difference to hold-current; tactical accuracy; strategic and headway/TTC UNAVAILABLE with reasons |
| `stack/scripts/build_hicap_cache.py` | §training-data | POD-side: factored vocabulary from TRAIN windows, per-clip cache from `build_episode` + Edge taps (not run) |

| `imagination.py` | §imag | Tier 0 soft outcome-aware targets + teacher costs; Tier 1 E-head (24 outputs, state only) + analytic expectation over mode AND onset; Tier 2 two cost heads, veto, learned benefit gate; composition with feasibility mask last (G1), veto not trigger (G4), confidence from the base score only (G5) |
| `elastic.py` | §elastic | 10 camera options (T0–T3), vision-only gate G-V, cost-sensitive full-information loss, asymmetric keep-alive + shadow mode, dropout curriculum 40/20/15/25, config embedding, L^cfg |
| `knowledge.py` | §vocab, §prune | text-defined nodes: versioned NodeLibrary (new token from its definition, ids never move), System-2 verifier (fires only when uncertain, scores never generates) |
| `distill.py` | §student | optional student: shared action tables, cos+MSE+KL distillation, decision-preservation margin condition |
| `strategic.py` | §vocab | strategic labels from the v8.1 release mapped to v9 rows around the 8 s anchor |
| `full.py` | whole | `HiCAPFull` (58.3 M) and `FullLoss` with every term active when its supervision is in the batch |

Not yet implemented: the jobs that PRODUCE supervision for the new heads from the corpus (exogenous descriptors, counterfactual costs and per-option errors from the obstacle-track and extra-camera caches; `extra cameras do not exist yet`), the text-tower embedding job for node definitions, real-data conformal calibration, the audio tower and event-metadata training, the audio/native-omni test. Those come with the experiments.
