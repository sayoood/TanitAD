# RESULT (interim, 2026-10-19): S0 provenance banked, SR-TTC R1 fitted at f = 10 %, EP-R1 gated and launched

Agent: Architecture & Inference, for the Master Mind. CPU only. Landed tip used for all code: `d58990cfc2fd3ddebeaf2a96197ca56ea5b73bf0` (origin/agent/arch-inf-20260803, verified with ls-remote). Staged, not committed.
This file is interim: the EP-R1 fit and its dev reading were still pending when it was written (see `C:/Users/Admin/par18/r1run/STATE.md`).

## 1. S0 provenance (MEASURED)
S0_DONE 2026-10-19 00:39, no S0_FAILED, `PARTIAL: false`, tokens 8,922/8,922, sets 15,808/15,808, 23 units, `logs_without_any_token` = 0, identity controls: fan-path logit max 0.0, set-path logit differs on 15,808/15,808. 160 json/log/tsv/marker files from `srttc_s0/run/paths_s0_results.json` staged under `raw/s0/` (no `*.sets.npz`; the 2 plan files already at the tip are byte-identical and were skipped). `S0_HANDOFF.md` line 25 corrected: `logits [S,64]` -> `logits [S,64,6]`, nothing else changed. NOTE: the post-step list includes 38 `.tsv` copy manifests (4.6 MB); drop them at landing if json/log only is wanted.

## 2. Launch gates (MEASURED, minted by the gates, never hand-written)
| gate | stage | result |
|---|---|---|
| `srttc_r1_gate.py` | run, f = 10 | PASS: G-COMMIT, G-DVB, G-PATH, G-BANK, G-LIVE, G-SUITE |
| `epr1_gate.py` | run, f = 10 | PASS 6/6 on the REAL manifest. This is the first real-manifest check (PREREG_EPR1 outcome A did not fire) |

## 3. SR-TTC R1, f = 10 % (fit-side; no dev or navtest number read)
`raw/r1/f10/r1_result.json`, sha256 `f447e71530e2fcc2a232ff47c06225efdd31d4200e6938d77d1d73a08ea2f37f` (570 s). c* = lr 1e-3, anchor 0.3, seed-mean pooled out-of-fold safe/violating concordance 0.70007, `criteria_agree` true. c* sits on the grid corner (lr and anchor both at their maximum). No grid-edge rule is registered for SR-TTC (grep of PREREG_SELRANK_ARM, amendments 1-2 and MM_DECISIONS_R1 for edge/corner/boundary: 0 hits); EP-R1 has one (row E1). Disclosed, no action taken. P2's in-sample reference is labelled IN-SAMPLE (MM_DECISIONS_R1 row 9) and is not the 0.72 of section 7. Evidence class MEASURED, fit-side, estimator: 5-fold grouped OOF, 2 training seeds. Not a result about H-SRTTC.

## 4. Pending
EP-R1 fit (launched 00:53), then C2, then EP-R1's dev gate once (E1 first). SR-TTC's dev gate waits for `srttc_dev.py`.
