<!-- Final report of a read-only survey sub-agent ("Survey refav1 training readiness"), transcribed
verbatim from the message it delivered to the coordinator on 2026-09-27. Its transcript file was empty
at extraction time, so programmatic extraction was impossible. Snapshot b3f7ea6f. Numbers are as the
survey reported them; the coordinator re-read the full-grid and p4 `lonshift` rows against their raw
files (see ../STATE_AND_GAPS.md §0). "REG" = Project Steering/MODEL_REGISTRY.md, "GC" =
Project Steering/GOALS_AND_CLAIMS.md, "MM6" = Project Steering/Decisions/2026-09-06-mm-decisions.md,
"AI/" = TanitAD Research Lab/Architecture & Inference/Research/, "†" = read with `git show b3f7ea6f:`. -->

# refav1 (REF-A v1): where it stands at tip `b3f7ea6f` (2026-09-26)

Integration escalations (each needs a Master Mind or PI decision): nine completed dev-box planner arms
were never read into the repo (the pre-registered A1 `kammshift` is one of them); the distance-keeping
planner hook was built and tested off-branch and is missing at the tip; the launch gate that CLAUDE.md
made mandatory today does not exist in the tree, and the gate as specified covers REF-C only; the
registry and the product ledger are stale for refav1.

## Executive summary
1. What it is: frozen DINOv3 ViT-L/16 fp8 features → WideAdapter → three token-field world models (0.2 s×30, 0.6 s×10, 1.5 s×4) linked by FiLM; a test-time iCEM planner (300×30, K=10 at 0.2 s) injects constant-velocity, hold and brake plans into its own candidate population; 182,459,701 params. One complete run, `refav1-b1-v72-ep3-speed` (21,109 steps, finished 2026-09-04), trained on B1/v7.2 (4,572 episodes) — not the parity corpus.
2. Last state: the last refav1 experiments were zero-training planner arms on 2026-09-05/06; the line has been parked since. The 2026-09-15 PI order is refcv6 → refav1 → v7.
3. Headline T1 read: on the full 282-window / 141-episode grid the shipped planner outputs an injected do-nothing plan on 282/282 windows. `cl−ha0` ADE +0.0158 [+0.0007, +0.0315], separated and worse; `cl−ha` and `cl−ha0_ext` ties; LON speed error +0.2706 m/s worse than hold-action. It does not beat doing nothing — the same verdict as refcv4b and refcv5-v2.
4. What training bought (T0 / representation, MEASURED): the world model beats "persist the last feature field" from 1 s (+0.1129) to 6 s (+0.2454); the LON tactical head reads 0.4326 vs a 0.2979 majority floor with nav withheld; the lead gap is decodable from the latent, its closing rate is not.
5. Diagnosis: the flat plan is a property of the cost, not the weights. A perfect goal makes the plan 2× worse — the search is healthy and the objective is misspecified. Lateral: the turn token commands κ=0.08 (R 12.5 m), ~5–10× sharper than the human's turns. Longitudinal: the goal token commands a≡0 on 29–31 of 40 windows. The terminal-speed weight `W_VEND` has never acted, because `target_speed` is never passed to the planner.
6. Best measured lever: `lonshift` (centred-cosine cost `ccos` + `W_KAPPA` 15.11 + `a0_shift`), measured only on the 40-window turn-dense p4 panel (8 episodes): ADE 0.7868 vs `ha0_ext` 0.8772 = −0.0889 [−0.2049, +0.0137] (parity; the seed replicate agrees); beats `ha0` on ADE, FDE and all three LON metrics, separated; still loses LON (+0.2535 m/s) and both tactical metrics to `ha0_ext`. Never run on the full grid.
7. Stranded results: nine finished dev-box arms and four or more Thor arms were never read back, including the best longitudinal point estimate, `loncomb3` (LON speed 0.4665 vs 0.5657 for `lonshift`).
8. Missing for a positive pre-registered result: a full-grid confirmation with an inference-seed replicate; PI rulings on `W_VEND`, lever defaults and the turn vocabulary; instrument fixes; and for any retrain the launch gate, code provenance and a closing-rate representation.
9. Cheapest path: read and bank the stranded arms (0 GPU) → pre-register → run the full grid ×2 seeds (~3–7 GPU-h) → PI ruling on `W_VEND`.
10. No closed-loop result and no HF upload for refav1 exist in the surveyed sources. Training variance has never been measured.

## 1. What refav1 is

| Item | Value | Source |
|---|---|---|
| Encoder | Frozen DINOv3 ViT-L/16, 640 tokens / 120° / 256×640, fp8 cache; ImageNet mean/std applied at encode | `AI/2026-08-18-refa-v1/REFA_V1_DESIGN.md:37-45`; `stack/scripts/dinov3_fp8_encode_ship.py:102-124` |
| World model | WideAdapter (no bottleneck); operative 0.2 s×30, tactical 0.6 s×10 (64 queries), strategic 1.5 s×4; strategic→tactical→operative FiLM chain | `REFA_V1_DESIGN.md:49-58` |
| Nav input | Injected into the tactical policy and the field predictors; the strategic subspace predictor stays nav-free; nav args (distance, time) discarded | `stack/tanitad/refs/refa_v1.py:905,1557-1664`; `GC:3577`; `GC:9195` |
| Planner | iCEM 300/30/30, β 2.5, horizon 10 steps × 0.2 s, baselines injected | `stack/tanitad/refs/refa_v1_plan.py:79-91,126` |
| Cost | Goal term + `W_JERK` 0.02 / `W_KAPPA` 0.05 / `W_VEND` 0.10 (`W_VEND` dead); turn token `GOAL_KAPPA_TURN` 0.08 | `refa_v1.py:116-119,433-435` |
| Params | 182,459,701 (checkpoint, M); the design's launch config was 174,043,172 | `REG:2089`; `REFA_V1_DESIGN.md:64` |
| Trainer argv (ep3) | `--target-space frozen --detach-aux-targets --bptt-truncate 15 --ema-targets --precision bf16 --tf32 --skip-nonfinite --speed-channel --min-participation 0 --bs 8 --lru 64 --seed 0` | `TanitAD Research Lab/Tools & DevEnv/Research/2026-09-07-thor-checkout-drift-a13/raw/thor_state_probe.txt:57-70` |
| Corpus | B1/v7.2: 168,910 windows over 4,572 episodes; labels md5 `0ff90213…` (train) / `aa12c948…` (eval); allowed by PI rulings | `REG:2092`; `GC:3325`; `GC:5735` |
| Ego input | Measured v0 as a third action channel; a missing v0 raises | `REG:2093` |

## 2. Runs and checkpoints

| Run | Outcome |
|---|---|
| `refav1-b1-v72-1ep-21109` | Collapsed at ~step 450; relaunched with frozen targets, retired at step 1,000; its directory name is not its step (`GC:47`) |
| `refav1-b1-v72-ep2-ema-bf16` | Retired at 6,850: speed never reached the model (`GC:5473`) |
| `refav1-b1-v72-ep3-speed` | Complete 21,109/21,109 (checkpoint mtime 2026-09-04 02:55). Thor showed no trainer and no supervisor on 2026-09-07 (`thor_state_probe.txt:24-49`). Copies: Thor `/home/nvidia/experiments/…/ckpt.pt` and dev box `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt`, both md5 `1189bc02…` (`AI/2026-09-05-refav1-lonshift-t1/raw/SHIP_VERIFY.md:36-41`); dev-box file exists at 2,122,997,633 B (metadata probe). No HF copy is mentioned anywhere in the repo |

Code provenance: ep3 was trained from Thor's older stack (its `refa_v1.py` 1,885 lines; the repo's 3,030), so a re-run from HEAD would not be a replication (`…/thor-checkout-drift-a13/RESULT.md:344,378`); the registry still does not record this (`Project Steering/BACKLOG.md:34`, A13c, open). The trainer saves a single rolling `ckpt.pt` and writes no `summary.json` done-marker (`stack/scripts/refa_v1_train.py:795-801`) — why the supervisor relaunched the finished run 80 times (`REG:2114-2123`). The only complete checkpoint, and so the current best, is 21,109.

## 3. Results that matter

| Result | Number | Class · tier | Variance the interval answers | Source |
|---|---|---|---|---|
| Plan degeneracy, full grid (282/141) | κ≡0 on 282/282; 2 distinct plans, both injected baselines | M · T1 | Structural, from the cost | `REG:2129-2151` |
| cl − ha0 | ADE +0.0158 [+0.0007, +0.0315], worse; LON speed +0.0308, worse | M · T1 | Episodes only (1 inference seed) | `taniteval/results/RESULT-refav1-21109-openloop.md†:398-402` |
| cl − ha | ADE +0.0083 [−0.0574, +0.0703], tie; LON speed +0.2706 [+0.2140, +0.3312], worse; LAT "better" only because a straight line beats a drifting held κ | M · T1 | Episodes | same file, `:413-422`, `:51-55` |
| cl − ha0_ext | ADE +0.0267 [−0.0285, +0.0813]; LON speed +0.2706 worse; TAC lon −0.1277 worse | M · T1 | Episodes | `GC:57` |
| `ccos` repair, full grid | ADE 0.7116 (compensated) / 0.9268 (naive) vs `ha0_ext` 0.5207: worse | M · T1 | Per-arm CIs | `AI/2026-09-05-refav1-ccos-eval/RESULT.md:100-107`; `GC:6383` |
| STRATEGIC | Route head is a nav echo: 1.0000 true / 0.4184 shuffled / 0.6383 zero (= majority class) | M · head read | Episodes | `RESULT-refav1-21109-openloop.md†:349-357` |
| Distance-keeping, full grid | cl 26.21 m / 4.69 s / 22.90 s ≈ ha0; n=87 | M · T1 | — | same file, `:318` |
| World model vs persistence | +0.1129 at 1 s → +0.2454 at 6 s, separated; −0.0442 at 0.2 s | M · T0 | Deterministic | same file, `:471-479` |
| LON tactical head, nav withheld | 0.4326 [0.3546, 0.5177] vs 0.2979 floor | M · head read | Episodes | same file, `:378` |
| Lead gap in the latent | R² 0.3632 vs pixels −0.0513; paired +0.4145 [+0.2018, +0.6120]; closing rate: every arm spans 0 | M · representation | 42 clip clusters | `AI/2026-09-06-refav1-perception-probe/RESULT.md:62-108,110-139` |
| p4, `lonshift` (40 windows / 8 clusters) | vs ha0_ext: ADE −0.0889 [−0.2049, +0.0137] (seed 1: −0.0978); FDE −0.3706 sep; LON speed +0.2535 sep, worse. vs ha0: ADE −0.1368 sep. vs ha: ADE −0.1005, tie | M · T1 | Episodes + inference seed + 2 GPUs; lever/floor ratio 12.0× | `AI/2026-09-05-refav1-lonshift-t1/raw/thor/pd_thor.md:16-21,47-50`; `RESULT.md:15-91` |
| p4, `best` | ADE 0.8838 vs 0.8772 (+0.0066; ADE seed floor 0.0607); 0 friction-circle violations, bought with turn_left recall 0/11 | M · T1 | Inference seed for the feasibility claim | `AI/2026-09-05-refav1-cost-geometry/RESULT.md:1010,1058-1069`; `GC:8688` |
| `W_KAPPA` sweep (p4) | Interior optimum at 15.11; curvature MAE on turn windows 0.04578 vs straight floor 0.05936 (−23 %) | M · T1 | Inference axis only; episode axis not separated for wk7 | `MM6:1831-1891`; `GC:8982` |
| Turn asymmetry (75 windows) | `W_KAPPA` takes turn_left 0.3667 → 0.0000 at both seeds; the sign is never wrong (153/153) | M · T1 | Episodes + 2 seeds | `AI/2026-09-05-turn-asymmetry/RESULT.md†:577-617` |
| Decision-rule A/B | ADE +0.8927 worse, 11.2× its seed replicate | M · T1 | Inference seed | `GC:6425` |

Does it beat doing nothing at T1? No. On the full grid it loses to or ties every floor, and hold-action wins LON outright. On the non-representative p4 panel the best arm beats constant velocity but only reaches parity with `ha0_ext` (which itself leaves the friction circle on 18.5 % of windows, `MM6:888-910`). The LEADERBOARD row carries only `ha0`/`ha` (`Benchmarks & Eval/LEADERBOARD.md†:72`).

## 4. The diagnosis chain (2026-09-02 → 09-06)

Eliminated: "early in training" (the epoch-end read is identical in shape to step 1,000, `REG:2150-2151`); the world model being lateral-blind (`GC:5217`, refuted); the steer/κ ×2.9 unit mismatch as the cause of the flat plan (a real contract defect, but converting units changes nothing here, `GC:5101`); goal-head accuracy (a perfect goal is 2× worse, `MM6:5-158`; the head fine-tune is "do not launch", `AI/2026-09-05-refav1-make-it-drive/MUST_WE_RETRAIN.md:9-40`); the decision rule (`GC:6425`); the seed ladder and the `ccosh` hold branch, both null (`AI/2026-09-05-refav1-cost-geometry/HANDOFF.md:136-140`); `goal_reach_s`, refuted at 0 GPU (`GC:7833`).

Localised: the cost form — the goal term is ~1e-7 against penalties of ~0.1, so 100 % of the iCEM population is excluded before the world model is consulted (`GC:71-73`); lateral — the goal vocabulary (0.08 is too sharp; only 28 % of real turns get a correctly signed goal, `MM6:1521-1535`; `GC:7057`); longitudinal — the goal token commands a≡0 (`MM6:268-285`); left turns — the `ccos` goal term pays ~0 for a left turn against +0.716 for a right turn (`AI/2026-09-06-wkappa-frontier/raw/TURN_ASYMMETRY_EXPLAINED.md†`).

Fixes that produced MEASURED gains (all default-OFF): `W_KAPPA` 15.11 (lateral accuracy); the Kamm friction cap (safety at no measured cost); `a0_shift`, the best lever, cutting LON speed error by −0.2283 m/s (~47 % of the gap). Failed their bar: `ccos` alone, the decision rule, the decoded-gap distance-keeping cost (BAR-2 12/21 vs 0.80: `AI/2026-09-06-refav1-decoded-gap/RESULT.md:19-39`). Unread: A3 (goal-conditioned cost), A4 (plus κ_turn 0.02; `GC:7806,7854` still say RUNNING), A1 `kammshift`, `loncomb3`, the `seambase`/`seamon` pair (the jerk-seam pair shows LON speed −0.2360 as a point estimate only, `MM6:1791-1830`).

## 5. Missing before the next run can be a positive pre-registered result (ranked)

| # | Missing item | Owner | Unblock / cost |
|---|---|---|---|
| 1 | Read and bank the stranded arms: `C:/Users/Admin/refav1_margin/p4out/rec_{kammshift,loncomb3,seambase,seamon,gkappa,bestlad,combined_seed1,wk15_ladder,lonshift_s1}.json`, single copy. The Thor arms (A3, A4, `T_grs*`, the `T_*` dumps) UNVERIFIED | Eval | 0 GPU, hours of CPU; Thor access for the Thor half |
| 2 | Full-grid confirmation with an inference-seed replicate. Every lever result rests on 8 clusters; the episode axis is the binding one (`GC:8982`) | Compute | ~3 h per arm on Thor (M: 10,898 s); ~1.6 h per arm on the 4060 (E: 20.96 s/window, `taniteval/tools/REFAV1_ARM.md:253-256`) |
| 3 | A pre-registration for #2: bars vs both `ha0_ext` and `ha`, four families, a lever/floor ratio, `ha0_ext` feasibility beside every comparison, a deliberate-regression arm | ArchInf | 0 GPU |
| 4 | PI rulings: arming `W_VEND` with target max(0, v0 + a0·T) is reserved to the PI (`GC:7311`, pinned at `stack/tests/test_steer_conversion_complete.py:449-490`); default levers are the PI's call (`…cost-geometry/HANDOFF.md:102-108`); also the turn-vocabulary magnitude and the unit cutover (`BACKLOG.md` R21/R26) | PI | A decision |
| 5 | The distance-keeping hook is absent at the tip (`dk_spec` 0 hits; `stack/tests/test_refa_v1_dk_hook.py:48-54` skipped); `plan_source` labels seed winners "cem"; `cost_fidelity` (design gate G1) has no caller; no test drives the arm tool with a lever flag; the trainer writes no done-marker | Code | Small |
| 6 | Any retrain needs: the launch gate (`stack/scripts/launch_gate.py` absent; `SPEC_REFCV7.md` has 0 refav1 mentions; CLAUDE.md:1286-1310); code provenance (A13c); a closing-rate representation (`MM6:2223-2232`); answers to the frozen-trunk advisory for refav1 (`BACKLOG.md:35`) | PI + compute | ~20 h per epoch on Thor (E: 3.35–3.5 s/step, `GC:5561`) |

## 6. How the design was supposed to be proven

Design gates G1–G5 (`REFA_V1_DESIGN.md:156-164`): G1 (`cost_fidelity`) never computed; G2 (planner beats constant velocity) FAILED (+0.0158 worse, separated); G3 (beats REF-A at 5k) never run, now inadmissible because the corpora differ (`GC:3325`); G4 (`coarse_fine_agree` ≥ 0.8): the step-21,109 record reads 0.4255, below the gate and never adjudicated (`taniteval/results/refav1-21109-openloop.json†:10845`); G5 (adapter std > 0.4) end-of-epoch value never reported. Later pre-registrations, as written: cost geometry — outcome (a) fired, its own prediction P1 refuted; turn asymmetry — outcome A; perception probe — PASS on the gap; decoded gap — BAR-1 PASS, BAR-2 FAIL, BAR-3 PASS, BAR-4 PASS; vocab-L3 — WITHDRAWN at the premise; tactical decoder (`Project Steering/PREREG_TACTICAL_DECODER.md`) and `GC:4869-4871` — unrun. Deliberate-regression arms were used (`chord` failed as it must; `a_sustain=−a0` was worse); the known-value controls held (X−X = 0).

## 7. Contradictions and staleness (the registry wins)
1. `products/P4-training-pipelines/REFA_V1_PROVEN_IMPROVEMENTS.md:179-184` says "a DESIGN… not a trained result"; last touched 2026-08-23.
2. `REG:2095` still says `ccos` is "IN PROGRESS"; it was delivered and refuted (`AI/…ccos-eval/RESULT.md:3`; `GC:6383`). Registry §2.4 has no `a0_shift`, turn-asymmetry or perception rows.
3. `CLAUDE.md:105` places the defect in the "cost geometry"; `MM6:1521-1535` moved the lateral defect to the vocabulary and `MM6:268-285` names a separate LON goal-token defect.
4. The "≈0.30 m seed floor" is cited to `D-REFAV1-SEED-GOAL-MISMATCH` (`CLAUDE.md:214-222`; `REG:3035,3321`); it was measured in `D-REFAV1-DRIVE-AB` (`GC:6425`: +0.2956 on 10 windows). Other panels read 0.0047–0.1035 (`MM6:1567-1573`).
5. `MM6:2186-2197` says the LON gap needs closing rate; but `ha0_ext` only holds measured (a0, κ0) (`taniteval/tools/refav1_arm.py:416-445`) with no lead information, so a missing closing rate cannot explain the gap to that floor (the survey's own analysis).
6. `GC:7313`, `:7417`, `:7715` still say QUEUED and `:7806`, `:7854` RUNNING for arms that have landed or been abandoned; the seed-replicate file the W_KAPPA-frontier `RESULT.md` §5 points to (`RESULT_SEEDS.md`) does not exist.
7. `stack/tests/test_refa_v1_a_sustain.py:148-150` claims the arm tool passes `target_speed`; its single `.plan(` call site passes none.
8. Runtime direction conflict: `REFAV1_ARM.md:253-256` measured Thor 1.8× slower than the 4060; the distance-keeping `RESULT.md` §7 says the 4060 is 1.5× slower.

## 8. Cheapest experiments toward a positive result

| Rank | Experiment | Cost | Expected effect |
|---|---|---|---|
| 1 | Harvest the 9 dev-box records, compute paired four-family deltas against their own seed floors, and score `kammshift` against its pre-registered bar (ADE ≤ 0.8892 and turn_left > 0; `AI/2026-09-05-refav1-close-the-gaps/SPEC_CLOSE_THE_GAPS.md:112-127`) | 0 GPU | Could make `loncomb3` the best config (point estimates ADE 0.7739, LON 0.4665; H) |
| 2 | Full grid, `lonshift` (+ `loncomb3`) × 2 seeds, pre-registered | ~3–7 GPU-h | H: LANE_KEEP is 244/282 windows, and the maintain stratum is where `a0_shift` beats the floor (−0.158 ADE on p4), so an ADE or FDE win over `ha0_ext` is plausible; LON likely still behind. Risk: `ADAPT_SPEED_FOR_CURVE` (78/282 windows) is an absolute target `a0_shift` does not shift (`refa_v1.py:457-490`) |
| 3 | PI ruling on `W_VEND` at target v0 + a0·T | Decision + small code + arms | H: the only lever aimed at the remaining LON gap, and it encodes the same law as the floor that wins LON |
| 4 | `wk7` + `a0_shift` on p4 | ~1 arm | Keeps left turns (wk7 reads 3/11) while testing whether `lonshift`'s ADE survives |
| 5 | Forward-pass probe of how the goal-latent cost responds to κ for matched left and right turns | Minutes | Locates the left-turn defect |

## Deliverable manifest (as delivered)
| Artifact | Where |
|---|---|
| This report | Returned to the caller only; nothing staged (read-only survey per the brief) — now banked here |
| `RESULT-refav1-21109-openloop.md`, `turnasym_RESULT.md`, `LEADERBOARD.md` | Scratchpad only: disposable `git show` copies of repo content, nothing original |

## Could not read or verify (as delivered)
Thor: whether the Thor arms (A3, A4, `goal_reach_s`, the `T_*` dumps) completed. Dev-box records: contents not read (existence and size only). GPU availability. Snapshot coverage (3,796 of 17,531 tracked files; † paths read with `git show`). No tests were run; pins read from source. G5's end-of-epoch value and any refav1 HF upload not found.
