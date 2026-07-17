# Fleet Review — 2026-07-17 (Sayed-directed)

> Full-fleet audit: every agent's deliverables graded, high-value work integrated into the main
> stream, backlogs upgraded with resource-mandated deep tasks. Directive: "analyze all agents,
> evaluate their deliverables, integrate the high-value ones, deepen the backlogs, and force the
> fleet to use our resources (4060, Colab CLI, eval pod, pods)."

## 1. Deliverable grades (3 independent review passes over 8 branches, ~15k stranded lines)

| Agent | Deliverable | Grade | Disposition |
|---|---|---|---|
| arch-inf | Blind-rollout σ-dissipation + attractor collapse (E1) + freeze-1 recovery | A− | MERGED; re-run on flagship@30k queued (their P0.1) |
| arch-inf | Stranded-orthogonality verification (withdrew own duplicate — exemplary integrity) | A | MERGED; D-021 claim corrected to "low-dim subspace" |
| prod-opt | H15 logvar-clamp NaN fix (witness-tested live bug) | A | **APPLIED to pod2 live stack + repo mainline same day** |
| prod-opt | Clean-GPU fp16 latency (93.7 Hz decision-safe; bf16 REJECT; ViT-only win) | A | MERGED; TRT-fp16 both-targets queued |
| prod-opt | REF-A↔flagship parity test (208-line controlled-baseline guard) | A | MERGED into test suite |
| bench-eval | Best-of-3 kinematic floor (speed-gated strata; CTRV 0.545 ≈ TanitEval's 0.544 — independent convergence) | A | MERGED; porting into TanitEval queued |
| bench-eval | Ridge ego-status ceiling + both-convention L2 + skill_score (AD-MLP repro) | A | MERGED; TanitEval rows queued |
| bench-eval | Behavior gate + `eval_behavior --config` fix + watch_gates | B (A nuggets) | Fix + decodability block lifted; pod-pull machinery superseded by TanitEval-on-pod |
| bench-eval | Cosmos robustness first pass (pixel-free suite, $0 acquisition) | B/A− | MERGED as pipeline validation; real numbers need CARLA |
| data-eng | D-016 R1 pinhole rectify (f_eff=266 by construction, observed_frac gate) | A | MERGED; calib consolidation queued |
| data-eng | PandaSet loader (fails-loud geometry, motion-heading yaw) | A | MERGED; real-bytes verify on pod3-idle queued |
| data-eng | OWN_DATASET_PLAN (license survey → owned-corpus strategy; ZOD = #1 ingest) | A | MERGED; ZOD is Data-Eng P0.1 |
| data-eng | Data Lake Phase-A (schema/ingest/view/shards + byte-equivalence gate REALLY tested) | A− | MERGED; run-at-scale + HF lake v0 queued |
| opponent | SC-13 stationary-lead (collision 0.0 vs 0.4) + W-09 first-responder (NHTSA directive) | A (oracle) | On tip; first ON-OUR-CHECKPOINT measurement queued (their P0.1) |
| tools-devenv | ci_gate one-command test gate | A | On tip; extension with merged suites queued |
| local uncommitted | validate_data.py (data pre-flight) / test_physicalai_rig.py (two-rig cy) | A / A | Committed this review |
| local uncommitted | eval_metric_rollout.py | B | Superseded by eval_grounded_rollout_4b + TanitEval; unify-or-retire |
| steering inbox | "External Anaysis.md" | B/C | Literature survey mis-filed as proposal → re-filed as landscape input; spot-verify before citing |

## 2. Integration executed this review

- **5 branches merged to tip** (arch-inf-0717, prod-opt-0717, bench-eval-0717, data-eng-0717,
  data-lake-phase-a): conflicts resolved (changelog-append pattern ×4; SCENARIO_DATABASE kept
  newer run-#3 side). ~15k lines un-stranded.
- **NaN insurance deployed:** logvar clamp [-10,10] applied to pod2's live stack (compile-verified)
  AND the repo mainline (`imagination.py` + `replay/arms.py` export). The running flagship process
  (~21k/30k) resumes on the patched build at its next restart; an immediate proactive restart was
  prepared but is **Sayed's call** (kill+resume loses ≤500 steps; declined-by-default until he
  confirms).
- **Resource Mandate written into `agents/_common-protocol.md`** (M-1/M-2/M-3 + new gate G-I):
  every run ≥1 real-compute experiment; resource declaration audited; eval pod opened to all
  agents; Colab job-card pattern; same-day merge rule (no stranded branches).
- **All six BACKLOGs upgraded** with a P0 fleet-directive section: deep tasks with
  goal/method/resource/falsifier, cross-referencing TanitEval + the imagination/planning panels
  + the review findings.

## 3. The combined picture (main stream + fleet, one story)

Flagship-speed @19k = **0.628** ade@2s (first CV-beater), CTRV oracle 0.544 still ahead; imagination
real but modest (vision_use 12.9%, imagination 8.7%); planning brains speed-starved (tactical wp
3.38 m) while goal-latent imagination is strong (cos 0.885). The fleet's stranded work slots into
exactly these gaps: arch-inf explains WHY recursion wastes imagination (σ-dissipation; freeze-1
holds), bench-eval hardens the denominator story (floors/ceilings/skill_score), prod-opt removes
the NaN sword over the remaining 9k steps and hands us deployable fp16, data-eng unlocks the
urban/turn data the 74%-straight corpus lacks, opponent supplies the scenarios the closed loop
will arbitrate with. **Nothing in the fleet contradicts the main-stream findings; three agents
independently converged on them.**

## 4. New ideas generated (combined plan, for DECISIONS/next runs)

1. **Flagship-v2 "imagination pack"** (arch-inf P0.3): v0+yr0→planning brains, future-action
   dropout, rollout-k≥12, goal-conditioned decode, nav dropout, TMS penalty, parallel-horizon
   H15 — every lever smoke-tested BEFORE the 30k verdict so the retrain starts same-day.
2. **Skill-score leaderboard** (bench-eval P0.1): every row reported as model ÷ per-stratum
   best-of-3 floor + ego-status ceiling — kills the shortcut-inflation class permanently.
3. **Owned lake v0 on HF** (data-eng P0.2): comma+cosmos+pandaset, one pull for every pod;
   ZOD next → the 74%-straight rebalance.
4. **Closed-loop sprint** (bench+opponent): CARLA-on-pod + SC-13/W-09/SC-06 scripts + the hub
   metric suite = D4–D6 arbitration; open-loop numbers stay "weak claims" until then.
5. **First on-our-checkpoint opponent row** (opponent P0.1): SC-13 via TanitEval windows — moves
   the competitive story from design-oracle to measured.
6. **NaN-class sweep** (prod-opt P0.4): grep-audit every unbounded exp/log/div on learned outputs;
   one witness test each — closes the silent-run-death class, not just this instance.

## 5. Ops notes (referenced by the mandate)

- **pod2 no-touch while flagship trains** (99% cgroup RAM; an eval OOM-killed it once already).
- **Memory-safe ckpt relay:** scp streaming read → `posix_fadvise DONTNEED` → verify trainer
  alive; never load weights into RAM on pod2.
- **Eval pod lock convention:** touch `/root/taniteval/results/LOCK.<agent>` for long jobs.
- **fsmonitor gotcha:** tool-written files can be invisible to git on this Drive repo — verify
  with `git status` after writes; toggle fsmonitor if needed.
