# W-A7 inputs (all located on the dev box; sha256 MEASURED 2026-11-14)

Frozen definition: `PREREG_P5C_A2.md` section 6 (copy at `../A2/`, repo commit `5a8c76d8`). Units: PDMS x 100.

| input | path | sha256 / blob |
|---|---|---|
| the 1,018 navtrain held-out rows (and each row's t20 = `labels20_t20_tc20_tw[:,0]`, banked P5-S Tc20s) | `E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz` (11 logs; the rows `p5c_labels.py control` reads) | sha256 `0a0866fe631a5b52ca9d42c88f46cc8944c1acd6d13bd0fcb0345c908a006072`, blob `6568837d` (= TF-1's) |
| bank trajectories | the dev-box bank copy (`C:/Users/Admin/p5c_scratch/bank`) was DELETED 2026-11-12. The rows' `traj` are taken from the npz above and written to `run/bank/targets_rank0.jsonl` (the labeller's input format). Cross-check recorded in the result: max abs difference of the dual labeller's Tc20s against the npz's banked Tc20s (A1 R-A bar 1e-9) | n/a |
| metric cache | **No navtrain harness metric cache exists.** Probed: `E:/Projects/TanitAD/data/` (none) and `F:/Archive/devbox-C/navsim/exp/` (only `metric_cache_navtest`, `metric_cache_navhard_two_stage`, `metric_cache_warmup_two_stage`, `metric_cache_smoke20`; no navtrain). `transp_score.py`'s own docstring says "NAVTEST ONLY (navtrain has no harness cache)". TF-1 and E2 rebuilt the cache parts per row from the local nuPlan DB with `navsim_pdm_targets.metric_cache_parts(scenario)`; W-A7 does the same (no `--metric-cache-root`). This is not a blocker: it IS the route TF-1/E2 used. | nuPlan DBs `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits` |
| banked TF-1 per-row tracked PDMS (control i) | `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf1/rows/tf1_s{0,1,2,3,4r}.jsonl`, field `Tc20s_tracked.pdms`; 1,018 rows, 0 errors, 1,018 distinct keys, all in the population | s0 `08953b30...`, s1 `c4a86c59...`, s2 `ac2d0085...`, s3 `bf1e4336...`, s4r `d6f68bb1...` |
| pinned dual labeller | `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/a1/p5c_dual_labels.py` | blob `50b9f0dbaa22563239d6c711e9f7dab846e1cc69`, sha256 `07e84b1c...` (= A1 Point 2) |
| `p5r_labels.py` / `p5r_run.py` | `.../scorer_gym_20261107/p5r/` | blobs `16d2aaf0...` / `d0a4084b...`, sha256 `84e1cac3...` / `58faf7a7...` (= A1 Point 2) |
| lambda* | `.../p5r/raw/lambda_star.json` | sha256 `83988dda...` (= A1; lambda* 3.0) |
| `p5_target_labels.py` | repo path `eval/p5_target_labels.py` in the E: tree | blob `294644df` (= A1) |
| A7 | `refe/planner.py` `repair_last_heading` (E: tree) | blob `6fb6598a` (= repo `5a8c76d8`) |
| scorer instrument | `instr/transp_score.py` (copy, `score_arms` only) | blob `3cbdd118fa82790caf7c38bdd8ce6f92320c575b` (= repo `5a8c76d8`) |
| scorer support | `refe/navsim_dac.py` `1884fee7`, `refe/navsim_pdm_targets.py` `f778e56e`, `refe/navtrain_scenarios.py` `848fe684` (E: tree = repo `5a8c76d8`); navsim `3e8291bf` at `F:/Archive/devbox-C/navsim/` | |

Deviation notes (none changes a threshold or a control):
1. `transp_score.py --a7` re-implements A7 inline (`p20[-1,2] = p20[-2,2]`) and does not call `repair_last_heading`, so per A2 section 6 it is NOT used. A7 is applied to the 20-pose label with `repair_last_heading` before `to_navsim`, as training's online labeller does.
2. The bootstrap is the TF-1 / E2 code verbatim (log-cluster, ratio mean, `default_rng(20260927).integers(0, L, (10000, L))`), made paired by drawing once and applying it to the per-row delta. A test pins it to the TF-1 formula.
3. Threads: every BLAS/OMP variable is 1 for the whole chain (brief), where the labeller's own default for Tc20s is OMP 2. Expect Tc20s to differ from the banked label at about 1e-14 (A1 note); control (i) is on PDMS within 1e-6.
