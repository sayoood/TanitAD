# Deliverable Manifest — Frozen-WM + Learned-Planner (Stream D `planner_on_frozen_wm`)

**Agent:** frozenwm-planner subagent · **Date:** 2026-07-23 (Europe/Berlin, UTC+2) ·
**Host used:** pod1 `tanitad-pod` (RTX A6000, FREE) · **Lock:** `gpu_lock.sh acquire frozenwm-planner`
(released at end) · **Additional direction — nothing running was modified** (v4.2b/pod2, Branch B/pod3,
REF-C weights, all untouched; used a read-only copy of v1's frozen checkpoint).

## Staged in the repo (this folder — `git add`ed, NOT committed, NOT pushed)

| Artifact | Path (repo-relative to the incoming folder) | What it is |
|---|---|---|
| Research bank (P0) | `RESEARCH_frozen_wm_learned_planner.md` | Frozen-WM + learned-planner literature (Dreamer/TD-MPC2/MuZero/PlaNet/IRIS), mechanism ranking for our setting, driving-specific (MILE/Think2Drive), frozen-ceiling theory. Primary-source cited. |
| Design bank (P1) | `DESIGN_frozen_wm_learned_planner.md` | Asset map, architecture (planner on frozen WM via the action channel), training recipe, the pre-registered discriminating experiment. |
| Experiment writeup (P2) | `EXPERIMENT_frozen_wm_planner_result.md` | Arm F/W/B result, adjudication vs pre-registration, honest limits. |
| Experiment writeup (P2 follow-up) | `EXPERIMENT_amortised_mpc_result.md` | Amortised-MPC prototype (coordinator follow-up 2026-07-24): CEM search + distilled prior, verdict + limits. |
| Experiment writeup (hardening) | `EXPERIMENT_40ep_hardening.md` | 40-ep clean-val hardening (coordinator follow-up 2026-07-24): oracle ceiling decision-grade on all 40; W data-limited on eval pod (no train corpus). |
| Experiment writeup (contender) | `EXPERIMENT_bigplanner_result.md` | Bigger feed-forward planner sweep (coordinator follow-up 2026-07-24, for Sayed's contender call): capacity 11× is FLAT (~0.60) → feed-forward architecturally limited, not a search-matching flagship. |
| Experiment writeup (crux) | `EXPERIMENT_valuemodel_result.md` | Learned value-model + CEM search (coordinator follow-up 2026-07-24, THE crux for the contender call): value-search FAILS (1.02, worse than feed-forward) → frozen-WM is NOT a contender, stays the fallback. |
| Raw result JSON (seeded, paired) | `artifacts/results.json` | Arm F/W/B: refs + 3 arms + paired episode-cluster bootstraps. |
| Raw result JSON (first run, unseeded) | `artifacts/results_run1_unseeded.json` | Provenance / reproducibility cross-check. |
| Raw MPC result JSON | `artifacts/mpc_results.json` | Amortised-MPC: search (cold/warm) + amortised prior + paired vs W/oracle. |
| Raw 40-ep result JSON | `artifacts/results_40ep.json` (+ `_s300.json`) | 40-ep hardening: oracle/CV/holdv0 on all 40 + W k-fold CV + paired. |
| Raw bigplanner JSONs | `artifacts/bigplanner_{v2,large,mlp}.json` (+ `.log`) | capacity sweep (3 files from run reordering): wplus/med · large · mlpbig/mlpwide, each vs W/oracle/search paired. |
| Bigplanner harness | `artifacts/big.py` | configurable planner (MLP/query head) capacity sweep on the frozen WM (reuses `run.py`). |
| Raw value-model JSON | `artifacts/valuemodel_results.json` (+ `value.log`) | learned-value CEM search: V-search vs W/oracle/GT-search paired + rank-corr diagnostic. |
| Per-window arrays (value) | `artifacts/perwin_value.pt` | per-window ADE for V-search + cold-GT + W/oracle (aligned). |
| Value-model harness | `artifacts/value.py` | collect (GT-CEM) → train V(state, plan)→cost → CEM search by learned V → paired + diagnostic. |
| Per-window arrays (40-ep) | `artifacts/perwin_40ep.pt` | Per-window ADE for W-CV + oracle/CV/holdv0 on all 881 windows. |
| 40-ep harness | `artifacts/run40.py` | Encode 40-ep val + controls + W episode-disjoint k-fold CV + paired (self-contained). |
| Per-window arrays (F/W/B) | `artifacts/perwin.pt` | Per-window ADE for every arm + oracle/CV/holdv0. |
| Per-window arrays (MPC) | `artifacts/perwin_mpc.pt` | Per-window ADE for search cold/warm + amortised prior + W/oracle (aligned). |
| Encode-cache script | `artifacts/encode.py` | Stage 1: frozen-encode train subset + val → per-frame state cache. |
| Training harness (F/W/B) | `artifacts/run.py` | Stage 2: train Arms F/W/B on the frozen WM, eval, paired stats. |
| Amortised-MPC harness | `artifacts/mpc.py` | CEM search over frozen WM + distil prior + paired (reuses `run.py` window ordering). |
| Mechanism smoke | `artifacts/smoke.py` | Validates the differentiable rollout + analytic-gradient path. |
| Run logs | `artifacts/{run,encode,mpc}.log` | stdout of the canonical runs. |

## Lives on the pod (reproduce/rerun; not copied into the repo — bulky / regenerable)

| Artifact | pod path | Note |
|---|---|---|
| Scripts (identical to staged) | `tanitad-pod:/root/frozenwm/{encode.py,run.py,smoke.py}` | canonical run copies |
| Frozen-state cache | `tanitad-pod:/root/frozenwm/cache/{train,val}/ep_*.pt` | 400 train + 12 val episodes, per-frame v1 states fp16. ~0.2 GB. Regenerable via `encode.py`. |
| Result + logs | `tanitad-pod:/root/frozenwm/{results.json,perwin.pt,run.log,run2.log,encode.log}` | copied into `artifacts/` |
| Frozen WM (read-only) | `tanitad-pod:/root/models/flagship-30k/ckpt.pt` | v1, NOT modified |

## Reproduce (on pod1, from the staged scripts)

```
gpu_lock.sh acquire frozenwm-planner 60 --ttl 21600
cd /root/frozenwm
PYTHONPATH=/root/TanitAD/stack python3 encode.py --n-train 400     # → cache/
PYTHONPATH=/root/TanitAD/stack python3 run.py --arms F,W,B --steps 3000 --out results.json
# amortised-MPC follow-up (needs perwin.pt from the run above):
PYTHONPATH=/root/TanitAD/stack python3 mpc.py --P 128 --I 4 --n-train-win 1000 --distil-steps 4000 --chunk 8192 --out mpc_results.json
# bigger-planner capacity sweep (needs perwin.pt + perwin_mpc.pt for the refs):
PYTHONPATH=/root/TanitAD/stack python3 big.py --arms wplus,mlpbig,mlpwide,med,large --steps 5000 --lr 4e-4 --warmup 300 --out bigplanner.json
gpu_lock.sh release frozenwm-planner
```
40-ep hardening (on the eval pod `tanitad-eval`, self-contained — encodes the 40-ep val in-memory):
```
gpu_lock.sh acquire frozenwm-40ep 60 --ttl 14400
cd /root/frozenwm40 && PYTHONPATH=/root/TanitAD/stack python3 run40.py --folds 5 --steps 1500 --out results_40ep.json
gpu_lock.sh release frozenwm-40ep
```

## Integration / escalation (per Agent Operating Standard rule 3)

- **This is an ADDITIONAL research direction, not a merge request into a running arm.** No integration into
  v4.2b / Branch B / REF-C is requested or implied.
- ⭐ **CRUX — learned value-model search (2026-07-24, `EXPERIMENT_valuemodel_result.md`):** the deciding test
  for the contender call. A learned value/cost model + CEM search (the deployable replacement for the
  GT-future cost) **FAILS: V-search 1.02, paired-WORSE than feed-forward W 0.599** — an aleatoric wall (V can
  only learn `E[cost|state]`, not this window's actual future) plus **adversarial exploitation** (CEM finds
  candidates that fool V). **The 0.132 GT-search prize is hindsight-privileged, not deployable planning.**
  **DEFINITIVE: frozen-WM is NOT a search-matching flagship contender by any tested route** (feed-forward
  0.60 · bigger 0.60 · distill 1.40 · value-search 1.02) — **it is a safe cheap ~0.60 FALLBACK.** This
  supersedes the amortised-MPC "product path worth pursuing" note.
- **Contender evidence — bigger feed-forward planner (2026-07-24, `EXPERIMENT_bigplanner_result.md`):** for
  Sayed's frozen-WM-as-flagship-contender call. Scaling the feed-forward planner **11×** in W's own head
  family is a **FLAT** line (W 0.599 → mlpbig 30.8 M 0.601 → mlpwide 42.6 M 0.599, none paired-separated);
  better recipe ties W (0.588); bigger query-decoder planners overfit worse (0.82–0.86). **Feed-forward is
  architecturally limited (~0.60), NOT a search-matching contender** — the W→search(0.132) gap is a
  test-time-search-vs-policy gap, closeable only with a **learned value model** (new arm), not a bigger
  policy. **Frozen-WM feed-forward stays a solid cheap FALLBACK (~0.60, paired-beats CV, WM undegraded).**
- **40-ep hardening (2026-07-24, `EXPERIMENT_40ep_hardening.md`):** the frozen-WM **simulator** is hardened
  decision-grade on the full 40-ep clean val — **oracle-action 0.4271** (= v1's canonical number, CI-separated
  far below CV). **W's deployable number could NOT be hardened on the eval pod** (it has no train corpus, and
  W is training-data-hungry: a 5-fold CV on ~700 windows/fold gives a data-starved ~0.95 that only ties CV) —
  **W's valid fallback stays the 12-ep 0.599.** ⚠️ **ESCALATION (one-step unblock):** a clean 40-ep W needs a
  train-episode cache on the eval pod; the **400-ep frozen-state cache already exists at
  `pod1:/root/frozenwm/cache/train`** (~0.2 GB) — copying it to the eval pod (out of scope here, pod1 was
  excluded by the brief) + one ~20 min train pass would produce the decision-grade 40-ep W.
- **Amortised-MPC prototype verdict (2026-07-24, `EXPERIMENT_amortised_mpc_result.md`):** CEM search over the
  frozen WM finds plans **4.5× better than arm W** (warm 0.132 vs 0.599, paired-separated) → **the product
  path is worth proposing to Sayed** (his go, per the pre-registration). **Critical guardrail measured:** the
  naive "distil the search into a fast action-prior" design **fails** (prior 1.399, paired-worse than W) — the
  hardened arm must use **test-time search + a learned value**, NOT a distilled action-prior. Full-arm
  commitment stays Sayed's call.
- **Related sibling streams (untouched, noted for the orchestrator):**
  `incoming/2026-07-23-planner-wm-gradient-coupling/` (fights the SAME degradation with gradient surgery —
  this bank is the freeze-instead alternative) and `incoming/2026-07-23-v4-fromscratch/`.

## Evidence class of the headline

`MEASURED` (ours) — raw at `artifacts/results.json` + `perwin.pt`, harness at `artifacts/run.py`, on
pod1 under `gpu_lock`, apples-to-apples with taniteval (reproduced CV 0.8463 vs registry 0.8377, hold-v0
0.7883 vs 0.7876, oracle-action 0.4045 vs registry full-set 0.4271). Scale caveat stated in the writeup:
**12-episode val subset** (the pod's val), not the full 40-ep eval-pod set.
