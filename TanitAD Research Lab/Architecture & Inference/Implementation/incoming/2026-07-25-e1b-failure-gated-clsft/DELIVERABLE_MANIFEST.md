# E1b — deliverable manifest

`2026-07-25` · every artifact and WHERE it lives. Repo copies are STAGED (git add), never
committed/pushed (Agent Operating Standard). Runnable copies live on `tanitad-pod3:/workspace/e1b/`.

## Docs (repo, staged)
| artifact | repo path |
|---|---|
| Pre-registration (both outcomes, metric, estimator, disjoint split) | `…/2026-07-25-e1b-failure-gated-clsft/PRE_REGISTRATION.md` |
| Substrate / P0 gate (MEASURED) | `…/2026-07-25-e1b-failure-gated-clsft/SUBSTRATE.md` |
| This manifest | `…/2026-07-25-e1b-failure-gated-clsft/DELIVERABLE_MANIFEST.md` |
| Launch confirmation (PID, log, check + eval cmds) | `…/2026-07-25-e1b-failure-gated-clsft/LAUNCH_CONFIRMED.md` |

## Code (repo `scripts/`, staged  ·  AND pod `/workspace/e1b/`)
| artifact | repo path | pod path |
|---|---|---|
| Failure mining (K=185 CL rollout on parity-train → recoverable pre-failure buffer) | `scripts/e1b_mine.py` | `/workspace/e1b/e1b_mine.py` |
| CL-SFT trainer (mined CL branch + open-loop replay, encoder frozen) | `scripts/e1b_clsft.py` | `/workspace/e1b/e1b_clsft.py` |
| Paired eval (FT vs base, run after FT) | `scripts/e1b_eval.py` | `/workspace/e1b/e1b_eval.py` |
| Launcher (mine → CL-SFT, detached-safe) | `scripts/run_e1b.sh` | `/workspace/e1b/run_e1b.sh` |
| P0 substrate probe | `scripts/e1b_probe_substrate.py` | `/workspace/e1b/e1b_probe_substrate.py` |
| E1a rollout harness (reused dependency; RESCUED — see SUBSTRATE §5) | `scripts/e1a_horizon.py` | `/workspace/e1a_e2a/e1a_horizon.py` |

## Run artifacts (pod only until the run completes; will be banked on the eval drumbeat)
| artifact | pod path | staged? |
|---|---|---|
| P0 probe result | `/workspace/e1b/probe_substrate.json` | copied to repo (below) |
| Mined failure buffer | `/workspace/e1b/mined_buffer.pt` (+ `.meta.json`) | meta staged post-run |
| FT checkpoint | `/workspace/e1b/refc-base-e1b-clsft/ckpt.pt` (+ config/metrics/train_log.jsonl) | banked post-run |
| Paired eval result | `/workspace/e1b/e1b_eval_result.json` | banked on eval drumbeat |

## Provenance copies (repo, staged)
| artifact | repo path |
|---|---|
| P0 probe result JSON | `…/2026-07-25-e1b-failure-gated-clsft/probe_substrate.json` |
| Mined buffer meta JSON | `…/2026-07-25-e1b-failure-gated-clsft/mined_buffer.meta.json` (post-mine) |
