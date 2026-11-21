# HANDOFF: FEAS2-T1 (H-FEAS2-T1-1), written 2026-11-02 16:48. The chain is RUNNING; a fresh agent finalises it.

Package `E:/Projects/TanitAD-artifacts/dagland/feas2_t1/`. Scratch `F:/TanitAD-scratch/feas2_t1/` (`tree_dag`, `tree_li` = KAPPA-T1 trees + the FEAS2 tool; `out/` = the 13 rollouts; `smoke/`). Nothing is in git (staged by nobody; the Master Mind lands).

## State
* Card frozen 2026-11-02T16:43:48+0100, sha256 `ecf9bb5d6b48595086656a470f69534d22d5be5d9882573504c1ab4df4ac77b2` (`FROZEN_SHA.txt`, which also pins the tool `4b66cc86...`, the reused Y files and the pre-check).
* Pre-check on seen KAPPA-T1 ticks (`raw/PRECHECK_ATTRIBUTION.json`, in the card): the heading harm tracks the curvature clamp (kbox = box), the accel clamp does not make it; the plan level shows no dip excess for the accel clamp to remove (M1 lives in the executed rows).
* Tool `code/v8f_t1_eval.py` (`--feasible off|box|kbox|abox|kabox`), `tests/test_v8f_t1_feas2.py` 16 passed (`logs/pytest_feas2.log`); real-checkpoint smoke (32 windows x 3 modes, `logs/smoke_*.out`): the reader's offline `lever()` equals the tool's plans, max abs 0.0.
* **Chain launched 2026-11-02T16:46:35+0100** through WMI (`feas2_t1_chain.cmd`, process 41468; python rollouts are its descendants). 13 rollouts, one at a time, 16 threads, ~150 s each: KA1 rerun `SREP2_DAGOACC_s0`, then DAG-OACC s0, s1, SOUP s0, s1 x kbox, abox, kabox. **Expected: rollouts done ~17:19, read done ~17:27.**
* Log `logs/chain.out` (markers `ZZFEAS2-*`), per-run `logs/run_<tag>.out`, read `logs/read.out`. Done markers `F:/TanitAD-scratch/feas2_t1/out/done_<tag>.txt`.
* **Markers:** `raw/CHAIN.done` = success (`raw/FEAS2_T1.json` written); `raw/CHAIN.fail` = a nonzero rc or a missing run, reason inside. RESUME = run `feas2_t1_chain.cmd` again via WMI (finished runs are skipped, an incomplete out dir is moved aside).

## Finalisation (a fresh Sonnet agent, no waiting on jobs: poll the marker file with one command, never a sleep loop)
1. If `raw/CHAIN.done` is absent and `chain.out` shows a RUN in progress, end and re-check later. If `CHAIN.fail`: read the reason; a REFUSED known answer is a result to report (KA1 = the fresh rerun differs from the reused Y: state the max abs deviation), not to patch around.
2. `RESULTS.md` from `raw/FEAS2_T1.json`: adapt `../kappa_t1/code/make_results_kappa.py` (same rows and outcome words). Differences: outcomes are `res["outcome"][mode][arm]` for mode in kbox / abox / kabox; rows are `res["arms"]["<CTRL|SOUP>_s<seed>:<mode>"]`; controls are KY (Y pins), KA1 (single rerun), KA2, KF, K3, KB1; `res["descriptive"]` holds the curvature partition. Also quote KAPPA-T1's box rows beside (`../kappa_t1/raw/KAPPA_T1.json`).
3. Check each outcome against the card's ESTIMATED table (kbox HARMS 70 %, abox NO EFFECT 65 %, kabox HARMS 60 %; M1 keeps D2 failing 85 %) and **execute the committed next arm for the outcome** (card, Rule Zero): HARMS on heading / S-POS -> `kbox2` = `project_feasible(a_max=inf, kappa_max=0.2, min_speed=2.0)`; M1 still separated-worse -> `lonfloor`. Each needs its own frozen card first.
4. `python -I code/make_manifest.py` (regenerates `MANIFEST.txt`), then hand back <= 300 words: outcome per arm and seed, the rows blocking DOMINANT, the next arm and its card sha.

## For the Master Mind (integration, PI)
* Land the package with `kappa_t1` (the reader imports `kappa_t1/raw/arrays/SREP_*.npz` and `F:/.../kappa_t1/out/SREP_*` read-only; the pins are in `FROZEN_SHA.txt`). The sha pin of `tests/test_v8f_t1_feas2.py` is the KAPPA-T1 tool `260bd175...`, supplied through `V8F_T1_KAPPA_TOOL`.
* Register the claim row `H-FEAS2-T1-1` (T1-kin; paired episode-cluster bootstrap, B 2,000, seed 0; episodes only) after the read. Not staged in git by this agent.
