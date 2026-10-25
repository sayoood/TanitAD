# tanitad.hicap — reference implementation of the HiCAP architecture (paper `Paper/HiCAP/hicap.tex`)

Status: **architecture, cached-feature dataset, 3-stage training loop and four-family evaluation implemented and unit-tested on CPU with synthetic data (32 tests); nothing trained on real data, backbone wrapper and pod cache builder not run (no GPU).**
Test: `PYTHONPATH=stack:taniteval python -m pytest stack/tests/test_hicap_architecture.py stack/tests/test_hicap_data_train.py`

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

Not yet implemented: agent-track cost targets (cost heads need the obstacle join), strategic labels (v7/v8.1 release), text-definition embeddings job, imagination tiers (E-head, Tier 2 critic), elastic-sensing gate, System-2 verifier, real-data conformal calibration. Those come with the experiments.
