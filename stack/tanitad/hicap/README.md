# tanitad.hicap — reference implementation of the HiCAP architecture (paper `Paper/HiCAP/hicap.tex`)

Status: **architecture implemented and unit-tested on CPU with synthetic data (22 tests); nothing trained, backbone wrapper not run (no GPU).**
Test: `cd stack && PYTHONPATH=. python -m pytest tests/test_hicap_architecture.py`

| module | paper | content |
|---|---|---|
| `vocab.py` | §action tree | v9 classes (3 lat × 7 lon = 21 cells, 19 populated), path/profile extraction, arc-length composition, farthest-point factored codebooks with a standstill sub-vocabulary |
| `model.py` | §backbone | motion adapter, cached-token projector (+age Fourier), nullable ego (v, a_lon, yaw rate — **no curvature**)/nav/route tokens, fusion (d 512, 6 L), level queries, gated audio (zero-init, bit-exact null), Matryoshka readouts, node towers, cost/refine/pair heads, class-routed MoE re-ranker; **56.4 M params** |
| `losses.py` | §scoring and training | level-wide soft-target critics, ancestor sums, masked/unmasked InfoNCE (H-HC4 ablation), consistency KL, goal loss, 3-stage curriculum, lr rule |
| `prune.py` | §pruning | R1 physical masks, exact pair feasibility, guard set, R3 route mask, R5 hysteresis, R6 class-conditional conformal + episode-blocked coverage |
| `decode.py` | §cascade | S0–S4 cascade and decision rule (batch 1 reference) |
| `backbone.py` | §backbone choice | Cosmos3-Edge tap wrapper (blocks 28/42/56, 160-token assertion, min_pixels fix) — **untested** |

Not yet implemented: the cached-feature dataset over the v9 arrays, the training loop and seeds, imagination tiers (E-head, Tier 2 critic), elastic-sensing gate, System-2 verifier, real-data conformal calibration. Those come with the experiments.
