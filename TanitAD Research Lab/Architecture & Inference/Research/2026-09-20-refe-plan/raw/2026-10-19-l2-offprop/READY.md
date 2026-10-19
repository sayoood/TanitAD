# READY - L2-OFFPROP: what the Master Mind runs after sealing

Build status: **BUILT, GATED (5 of 5 PASS on the selected recipe), UNSEALED.** The MM rulings E1-E7 and the 14:10 addendum are applied in code and tested. **No real arm (H, H0, H-null) has been trained and no unplanted A1-OOF has been read from real labels.** Every command below is CPU only (`CUDA_VISIBLE_DEVICES=-1`), no pod, no GPU lock.
Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`. Code: `code/` (LF sha256 list `code/MANIFEST.sha256`). Features: `D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz` (manifest `raw/feats_manifest.json`).

**Selected recipe: `lam10_lr0.001_ep15_prox0.3`** (`raw/recipe_chosen.json`; selection table `RECIPE_SELECTION.md`): lam_bce 10, lr 1e-3, 15 epochs (~570 steps per fold), prox 0.3, batch 256, Adam. **C-REACH A1-OOF (planted arm): 0.9984 seed 0, 0.9992 seed 1** (bar 0.98).

## 0. Read this before sealing (three findings, none of which I redesigned around)
1. **One unit test FAILS and was NOT loosened:** `test_semi_synthetic_hits_its_two_calibration_targets_and_reads_no_real_label` (A1(H) > A1(P2) + 0.02 on a 72-token planted world). Under the selected recipe the margin is **-0.0156** (gain +0.0044); under the source default it is -0.000036 (one pair short; that was the chain's "exactly +0.020"). Cause: the selected recipe is in-fan-weighted by design, so on 57 training tokens it barely moves; a recipe that finds the signal there degrades in-fan ordering. A companion regression test on the REAL tokens (sigA, beta 0.5) passes (gain +0.045). Expected `pytest test_l2.py -q`: **`1 failed, 34 passed`**. Detail: `RESULT_GATES.md` (KNOWN FAILURE), `raw/planted_margin.json`.
2. **L2-ADMIT's A1 >= 0.80 may be out of reach for any recipe that satisfies the SELECTION in-fan clause (-0.01).** In the semi-synthetic worlds (planted beta 0.5 signal, P2 at 0.735 / 0.785) NO feasible setting reaches 0.80 in either world: the best feasible reach 0.788 (sigA) and 0.792 (sigB), the pick 0.780 / 0.790. Settings that do exceed 0.80 in sigA (up to 0.820) break the clause by -0.019 to -0.125 in-fan (closest: `lam10_lr0.001_ep50_prox0`, sigA 0.8004, margin -0.019). Note the OUTCOME's own in-fan margin is -0.02 while the selection used the ruling's -0.01. Whether the real signal is stronger than beta 0.5 is what the real run measures; if it is like sigA/sigB the expected outcome is **L2-PARTIAL, not L2-ADMIT**. (`RECIPE_SELECTION.md` table.) The MM's rule applied literally; flagged, not redesigned.
3. **The two training seeds are near-replicates** (E8): H's A1-OOF differs by 0.0006 between seeds 0 and 1 on sigA (`DESIGN_CHOICES.md` section 5). "Both seeds" is therefore not independent evidence about training variance (H-ESTIM-SEED-1 stays open); the independent unit is the log fold structure.

## A. The rulings as built (each has a test that can fail)
| ruling | what the build does | where |
|---|---|---|
| **E1** null = GLOBAL shuffle | `permute_pert_labels` default and `ArmSpec.null_group` default are `global`; cells are `Hnull-global_r{100..104}_s{0,1}`; the `anchor` (withdrawn) and `token` nulls exist only as `--null-group` diagnostics and the read flags them "DIAGNOSTIC, not the sealed outcome". VOID = >= 2 of 5 global-null replicates above P2's A1-OOF + 0.05, per seed (E4 strict) | `l2lib.py`, `run_l2.py`; `test_the_default_null_is_the_global_shuffle`, `test_real_run_cells_are_the_arms_of_the_amended_prereg`, `test_read_uses_the_pooled_a1...` |
| **E2** A1 = POOLED | `summarise` reads `ev["all"]` (den 9,888; C-ID reproduces 0.6851 exactly); lon and lat are reported in every table and in `outcome.inputs`, ungated | `test_summarise_a1_is_the_pooled_quantity` |
| **E3 / E4** strictest defaults | `--null-pool all`, `--void-scope any_seed` | `outcome.py`, 2 mutations |
| **in-fan clause -> H0** | new arm H0 = the same recipe with the pairwise weight 0 (same init, folds, seed, minibatch stream; `fit_head(w_pair=0)`); outcome: in-fan(H) >= in-fan(H0) - 0.02 at the same seed; P2's in-fan is reported (`infan_p2`), never gated. Reach proven: with `w_pair = 0` the EP-row gradient equals `lam_bce * BCE` alone bit for bit; flipping every pair label leaves H0 bit-identical; a zero-step H0 is P2 bit for bit | `test_arm_h0_pairwise_term_is_dead...`, G-LIVE `h0_lever_reaches`, C-ID, `test_outcome.py` (31 scenarios, 27 textual mutations: the clause dropped / compared with P2 / `>=` to `>` / sign flipped are all RED) |
| **recipe** | re-selected with C-REACH as a HARD constraint (planted feature, both seeds); rule and table in `RECIPE_SELECTION.md`; `prox` (toward P2's head) added as the MM allowed; the recipe is a data file the run refuses to start without | `design_sweep.py`, `finalize_recipe.py` |
| **E6 -> C-LEAK structural** | `oof_scores` calls `fold_split` on every fold: asserts train/held-out token and log disjointness AND that every index array the optimiser consumes references only training tokens; after the folds every token must have been scored exactly once. No off-switch (the old `check_leak` is gone). RED mutations: ONE held-out token injected (all 5 folds), a whole fold, a held-out row routed into the pair or in-fan index arrays: each fires | `l2lib.fold_split`, `gates.py leak` (real tokens, SHUFFLED labels), 4 unit tests |
| **E7** not a launch-gate case | unchanged. Covered by G-LIVE, C-ID, the seal guard. **Integration of an L2 head into REFe's scorer (Step B) needs a `launch_gate.py` profile with G-HYG / G-DVB / G-LIVE entries for the new flags (`prox`, `w_pair`/H0, `--null-group`)**: a Step B launch blocker, not an L2 one | - |
| seal guard (new) | `run_l2.py` refuses unless every gate PASS was earned on THIS recipe file and THIS code (stamp = LF-normalised sha256 of the recipe and of `l2lib.py`, `gates.py`, `synthetic.py`) - the chain's C-REACH had been earned on a recipe later replaced | `test_seal_guard_refuses...`, `test_gate_stamp_is_line_ending_insensitive` |
| run path end to end | `run` + `read` on synthetic data: 15 cells, resume, INCOMPLETE on a missing cell, refuses a stale gate | `test_run_and_read_end_to_end_on_synthetic_data` |

## B. Open for the MM to write into the sealed prereg (build defaults shown)
| # | clause | build default |
|---|---|---|
| E5 | in-fan pair set | the SELRANK instrument (`selrank_attr.py:461-475`): P2's own top-8 by its aggregate, ordered pairs with dEP > 0.05 (strict), ties count half; the SAME top-8 sets for every arm |
| R1 | recipe rule details the ruling does not state | lambda = pairwise / in-fan weight = 1 / lam_bce; tie width 0.005; tie order: larger lam_bce, larger prox, fewer steps, smaller lr (here all four tied settings have lam_bce 10, so the lambda convention does not change the pick); in-fan clause applied in the two SIGNAL worlds, the null world REPORTED (A1(H) minus P2 +0.020, global-null max minus P2 +0.0001) |
| E8 | seeds {0, 1} | as drafted (near-replicates, finding 3) |

## C. Seal
1. Copy `DRAFT_PREREG_L2.md` to `PREREG_L2_OFFPROP.md`; carry `MM_RULING_L2_BUILD.md` (E1-E7 + addendum) verbatim; settle section B; add the sha256 of `code/MANIFEST.sha256`, of `raw/recipe_chosen.json` and the recipe key; delete the line containing `Status: DRAFT`; add `Status: SEALED <date>`.
2. `sha256sum PREREG_L2_OFFPROP.md` -> `$SHA`.
3. Re-check the build (about 1.5 min): `cd code && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 python -m pytest test_l2.py -q` (expects `1 failed, 34 passed`, the known failure) and `python test_outcome.py` (expects `ZZOUTCOME_MUTATIONS OK 27 mutations`). Gates already on disk: `raw/gates/*.json` (all `"pass": true`); to re-earn one: `python gates.py <cfeat|cid|glive|reach|leak>`.

## D. Run (15 cells = P2 + 2 x H + 2 x H0 + 10 x global H-null)
Measured cost of the whole run with H0 included: **~2.3 min on ONE process** (REAL tokens + synthetic labels, `raw/cost_semi.json`) / 2.5 min (`raw/cost.json`), 1 torch thread, box at 60-90 % CPU from other jobs. Cells are independent and resumable.
```
cd <repo>/"TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-l2-offprop"/code
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
G=../raw/gates ; R=../raw/recipe_chosen.json            # or C:/Users/Admin/par18/l2/out/gates and D:/Projects/TanitAD-artifacts/l2_offprop/recipe_chosen.json
PRE=<path of PREREG_L2_OFFPROP.md>; SHA=<its sha256>
python run_l2.py run  --prereg "$PRE" --prereg-sha256 $SHA --gates-dir $G --recipe $R --threads 2        # all 15 cells (or --cells P2,H_s0,H_s1,H0_s0,H0_s1 per process)
python run_l2.py read --prereg "$PRE" --prereg-sha256 $SHA --gates-dir $G --recipe $R                    # -> D:/Projects/TanitAD-artifacts/l2_offprop/run/L2_RESULT.json + L2_RESULT.md, prints ZZREAD_OK <outcome>
```
Defaults: `--feats D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz`, `--out-dir D:/Projects/TanitAD-artifacts/l2_offprop/run`. `run` writes `run_config.json` (recipe, code sha256s, feats sha256, argv). `read` applies `outcome.l2_outcome` clause by clause; an incomplete run reads `INCOMPLETE`, never a verdict. Diagnostics (not the outcome): `run ... --null-group anchor` (the withdrawn definition), then `read ... --null-group anchor`.

## E. What the outcome licenses (draft section 6)
No capability claim and no navtest look. A1-OOF on 24 SEEN logs licenses only the fresh-carve Step A re-run (`L2-ADMIT`), `L2-BRANCH` (`L2-PARTIAL`, `L2-FLAT`), or an instrument fix (`VOID`). Stamp every number SEEN-SCENE, SELECTION tier; the interval answers "another draw of LOGS" only. STRATEGIC is PAUSED (PI ruling 2026-09-19, ITEM 25) and is reported as PAUSED, never as a pass.
