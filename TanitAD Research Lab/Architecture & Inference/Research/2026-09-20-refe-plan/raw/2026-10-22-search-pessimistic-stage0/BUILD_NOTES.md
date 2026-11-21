# BUILD_NOTES: SEARCH-PESSIMISTIC Stage 0 (build agent, 2026-10-22)

**State.** Stage 0 is built, tested on SYNTHETIC data, and drafted. **It has not been run on the real tokens, and no IN/OOF statistic exists for any real token.** The run needs the MM to seal `DRAFT_PREREG_STAGE0.md` first (the script refuses otherwise). Nothing was staged, committed or pushed; no git index was touched; no GPU, pod or network was used.

## 1. Package (`C:/Users/Admin/par18/spess0/`)

| file | what | copies |
|---|---|---|
| `DRAFT_PREREG_STAGE0.md` | the pre-registration draft (folds, measures, estimator, controls, GAP / NULL / INVALID with numbers, power) | ONE (dev box) |
| `code/stage0.py` | the instrument; CPU, numpy only; `run`, `constants`, `check-inputs` | ONE |
| `code/test_stage0.py` | 56 tests on synthetic data (controls with their RED arms, estimator, decision rule, reach of every constant, loader and CLI path, and that the prereg draft quotes the code's constants digest) | ONE |
| `code/synth.py`, `code/geometry_real.json`, `code/extract_geometry.py` | synthetic worlds with the real geometry (log names and token counts only; no label) | ONE |
| `code/power_sim.py`, `code/mutation_census.py`, `code/rehearsal.py` | power and false-positive simulation; 38 textual mutants; whole-pipeline dress rehearsal | ONE |
| `run_stage0.cmd`, `run_stage0.sh` | the real read in one command; writes `raw/RESULT_STAGE0.json` | ONE |
| `raw/` | `power_sim.json`, `MUTATION_CENSUS.json`, `pytest_final.txt`, `rehearsal_final/*`, `check_inputs.json` | ONE |
| `code/MANIFEST.sha256` | sha256 (LF-folded) of every code file at hand-back | ONE |

**To run (MM):** (1) review §11 of the prereg; (2) copy to `PREREG_STAGE0.md`, compute `sha256sum PREREG_STAGE0.md | cut -d' ' -f1 > PREREG_STAGE0.sha256` (hash the file as stored; any later edit voids the seal); (3) `run_stage0.cmd` (or `.sh`). It runs the test suite first and aborts if red, then checks `CUDA_VISIBLE_DEVICES=-1`, the prereg's sha256, that the prereg quotes this code's `constants_sha256`, and the sha256 and size of all seven inputs. Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (override with `TANITAD_PY`); inputs: `D:/Projects/TanitAD-artifacts` (override with `TANITAD_ARTIFACTS` if the drive letter flips).

## 2. Inputs (MEASURED sha256 and size, 2026-10-22; equal to the banked manifests)

| key | path | bytes | sha256 |
|---|---|---|---|
| table | `D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz` | 13,304,426 | `8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b` |
| prehead | `D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/prehead_features_p2_heldout.npz` | 51,442,644 | `e37b9c554bf2c2546e958ff763dca23b71e0b77036a271d6a003c8ba2672f522` |
| feats | `D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz` | 67,074,052 | `49f6a034464580ef6e14bae9dc397b898a360bc9d7051d6d0676acdba141e616` |
| head | `D:/Projects/TanitAD-artifacts/l2_offprop/p2_score_head.npz` | 7,142 | `4b64c3feb89ca571228ed8d4cfa795f3ec5143e71d06545e287e0b9b13ff55d0` |
| bundle | `D:/Projects/TanitAD-artifacts/l1_stepA_run/bundle/bundle.npz` | 8,309,154 | `77e8584d8a02ee7a93b9239e1e33a16edade75689f36e1f32959960d0e2eb5c4` |
| perts | `D:/Projects/TanitAD-artifacts/l1_stepA_run/perts.npz` | 18,972,106 | `40ecafc88d401463767d74af5d41809d73b36009083e19d62e197f4bfd6bb8d5` |
| labels | `D:/Projects/TanitAD-artifacts/l1_stepA_run/labels.npz` | 2,151,734 | `7c986dcc736a6dcaf49dfaccd33f3ded9e9c4f2ec49b88f28bb728a71ebf9b27` |

Not used but named in the memo: the L2-OFFPROP code. Stage 0 re-implements the refit (numpy, analytic gradients) so that it needs no torch; `test_stage0.py` pins `aggregate` and the A3 concordance to L1's `stepA.py` (`C:/Users/Admin/par18/l1run/code`, those two tests skip if the file is absent) and Adam to `torch.optim.Adam` (skips without torch).

**Everything done to the real files:** sha256, shapes, counts, the loader's integrity assertions (token/log order, Step A alignment, logit and anchor-label equality between table and bundle, the stored provenance hashes, finiteness, `pdms` on `[0,1]`) via `stage0.py check-inputs` (all passed, `raw/check_inputs.json`), and the fold and half assignment (token counts only). Nothing else.

## 3. Results of the build

- **Tests:** `pytest code/test_stage0.py`: **56 passed in about 48 s** (`raw/pytest_final.txt`).
- **Mutation census** (`raw/MUTATION_CENSUS.json`): 38 textual mutants of `stage0.py`, run against the suite; **38 of 38 killed**. The suite initially let 5 mutants survive (a pairwise-label sign flip, misaligned Step A features, Adam without bias correction, scoring the wrong half, a swap-null that is always 0); each led to a test or a structural check, listed in §5.
- **Power / false positives** (`raw/power_sim.json`): 8 planted strengths × 24 worlds with the real geometry: 0 of 24 GAP verdicts at κ = 0; the smallest M1 gap detected at ≥ 80 % is 0.016 (prereg §8).
- **Dress rehearsals** (`raw/rehearsal_final/`, the whole `run_all` on 7 full-size synthetic worlds, final code): κ = 0 (seeds 61–64) **NULL (UNDERPOWERED)** ×4; κ = 0.5 NULL; κ = 1 **GAP**; κ = 2 **GAP**. Every control passed in all 7. M3 was uncertified in 4 of 7 (its shuffle control did not reach 4·SE ≤ 0.35 on the synthetic regret noise), which is the designed behaviour.
- **Synthetic null worlds, earlier in the build** (not in the final record, code since changed): 6 further κ = 0 worlds gave NULL.

## 4. ESTIMATED runtime of the real read

- **MEASURED on a full-size synthetic world with the real geometry (one process, 24-core box):** `run_all` 98 to 112 s; about 100 to 115 s wall including start-up. Under contention (7 worlds plus a census in parallel) 245 s.
- **ESTIMATED for the real read:** **about 2 to 3 minutes of CPU on one core-equivalent** (about 0.05 CPU-h), plus about 46 s for the suite that runs first, plus about 5 s to hash and load the 160 MB of inputs. Peak memory ESTIMATED under 2 GB (largest arrays: 112 MB features, a 223 MB float64 copy in the planted-label generator). No GPU.
- The memo's "≤ 1 CPU-h" holds with a wide margin.

## 5. What the build found (these changed the design; the MM may want them in the register)

1. **Pooled shuffle (control C-I).** A within-token shuffle of fan labels and perturbation labels *separately* does not put A3 at 0.5 for a blind score, because a perturbation is usually worse than its anchor (the two sets are not exchangeable). A z of 3.2 on M2 in the first rehearsal exposed it. Fixed by pooling the 160 labels of a Step A token and re-dealing; pinned by `test_pooling_is_what_puts_A3_at_chance`.
2. **M3 is too noisy to gate controls.** Regret flips with every pick, so M3's shuffle control needs 100 shuffles to reach 4·SE ≤ 0.35 even in a quiet world, and C-III on M3 failed on noise in a null world. Resolution: an uninformative M3 control **uncertifies M3** (it cannot trigger GAP or power a NULL) instead of voiding the read; C-III gates on M1 only.
3. **Strict "SEEN beats OOF" is not reliable on the real heads.** On synthetic null worlds the token-level overfit of a 257-parameter row on about 44,000 points ranges from −0.001 to +0.0125, so a strict `> 0` rule voids about 1 valid read in 30 (one of 36 worlds). The brief's control (ii) is therefore **gated strictly in the PLANTED world** (where a gap exists by construction) and, on the real tokens, as **non-reversal** (token-level z > −4). This is a deviation from the literal brief: it is flagged in prereg §11.
4. **A larger planted effect is not always easier** (labels saturate, the swap-null grows), so C-II-b passes on any κ ∈ {0.5, 1, 2, 4}.
5. Mutation survivors fixed: a test that re-implemented Adam instead of calling `adam_update`; no test of Step A pairing (now a constructed world on which P2 scores every pair correctly, read at 1.0 exactly); no test of the pairwise-term sign (now a world orthogonal to P2's rows that only a correctly-signed pairwise term can learn); no check that the tokens judged are search-half tokens.

## 6. UNVERIFIED (read before sealing)

1. **The recipe on rows NC, DAC, TTC, C.** R0's budget comes from L2-OFFPROP's EP-only sweep (INHERITED, semi-synthetic labels). Its behaviour on the other four rows of the real head is untested.
2. **Real-data noise.** Every SE, the M3 certificate, the C-I informativeness margins and the C-II-b planted gap are measured on SYNTHETIC worlds. In particular **the planted effect on the real features may not be read at any κ ≤ 4** (real features are low-rank and anisotropic; the generator standardises the planted direction, but the learnability is not known). If so the run is INVALID at C-II-b and the fix is in prereg §7.
3. **Δ = (0.01, 0.02, 0.35)** is judgement (M3's is derived).
4. **Sessions with a single segment.** 18 of 21 sessions have one segment, so IN shares the *segment* with the search token; P2's real exposure did not. A GAP cannot separate session from adjacency (prereg §9).
5. **Power for M2 and M3 on the real data** is not known until the read prints its achieved SE; the inherited M3 range (0.17 to 0.61 points MDE80) rests on the memo's banked half-widths.
6. `jsonable` renders NaN / inf as strings; an `n/a` measure (M2 with no Step A token in the search half) would appear that way. Not expected: the real design has 201 Step A search tokens.
7. The suite takes about 46 s and is part of `run_stage0.*`; a flaky test would abort a sealed run. No test uses wall-clock or a non-fixed seed.

## 7. Memo discrepancies to fix in the register

- "about 850 search-half tokens *t* of fold g" is the **pooled** search total (845); per fold it is 167 to 171.
- "trained on about 1,364 tokens" is four **full** folds; a model trains on the **fit halves** of four folds, about 690 tokens (about 44,000 fan points).
- Fold sizes with greedy balancing are **338 / 340 / 344 / 337 / 346**, not "about 341 each" (close).

## 8. Integration and decisions (headline)

1. **MM: seal the prereg** (after §11 of it) so the run can start. Nothing else blocks it.
2. **PI/MM decisions inside the prereg:** the three Δ, R0 (prox 0) as the gating recipe, GAP through any one certified measure, and the literal-versus-planted gating of control (ii).
3. After the read: update `D-REFE-SESSION-EXPOSURE-1` and the C2 `RESULT.md` wording ("sessions P2 was trained on"), as FEASIBILITY.md §7 already lists.

## 9. Deliverable manifest

| artifact | where | copies |
|---|---|---|
| every file in the §1 table | `C:/Users/Admin/par18/spess0/` | **ONE** (dev box; the MM lands it) |
| the memo it implements | `C:/Users/Admin/par18/spess/FEASIBILITY.md` | ONE (previous package) |

Nothing is staged, committed or pushed (MM override of rule 1). The package is the only copy.
