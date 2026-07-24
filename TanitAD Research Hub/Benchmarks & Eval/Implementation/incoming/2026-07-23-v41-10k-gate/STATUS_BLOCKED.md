# flagship v4.1 — G1 (10k) KILL-gate: **BLOCKED / INCOMPLETE — escalated, NOT decided**

`2026-07-23 (00:5x local / 22:5x UTC)` · eval host intended: pod1 (`tanitad-pod`, idle). **No eval was run on pod2** (v4.1 is training there). Evidence class on every claim: **MEASURED** (ours + artifact) · **INHERITED** · **HYPOTHESIS**.

## Headline

**The flagship's first KILL-gate cannot render a decision-grade verdict, and it must not be forced.** `run_gate.py check` returns **`BLOCKED`** — *"no --eval-json; the primary gate is a HELD-OUT milestone metric"* (artifact `v41_10k_gate_result_BLOCKED.json`). The blocker is **not** a missing cache — it is that **the v4 held-out eval harness and 4 of the 8 KILL secondaries were never built**, and pod1 is not provisioned. This is exactly the "state the blocker and escalate; do NOT eval on pod2 as a shortcut" case the brief named. Standing up brand-new, unvalidated eval instruments and using them to RESTART/REFUTE the flagship would violate GATE_PROTOCOL and V4_DESIGN §17 O-03 ("a gate that would kill the arm on a technicality").

**One concrete blocker removed:** the pre-registered gate card was **stranded** (referenced everywhere, never staged). I materialized it verbatim from the design-doc pre-registration and staged it → `Project Steering/Gates/flagship-v4.card.json`.

## What the gate needs vs. what exists (all MEASURED against code)

Card (`flagship-v4.card.json`): primary **`ade_0_2s ≤ 0.60`** (held-out taniteval, planner path, produced goal, via `cluster_bootstrap`) + **8 KILL** secondaries + 5 report-only.

| Gate input | Built & runnable for a v4 ckpt? | Evidence |
|---|---|---|
| **PRIMARY `ade_0_2s`** (held-out cluster-bootstrap) | ❌ **NO** — no v4 eval driver | No script loads a `flagship_v4` ckpt + runs its planner/produced-goal path over val to emit a `windows_<key>.pt`. `eval_flagship_v16.py` STRICT-loads `FlagshipV15Head`/`V15Config` → incompatible with v4's factorised head + `V4Config`. Glob `*v4*eval*` = none; only importers of `FlagshipV4Head` are the trainer + tests. |
| `wm_canary_ade_2s ≤ 0.55` (KILL) | ✅ in-loop only | Emitted to `train.log`/`metrics.json` by `train_flagship_v4.py:404-436,651`. **step 10000 = 0.45994 (PASS-direction), but see caveats.** |
| `oracle_in_fan ≤ 0.30` (KILL) | ⚠️ in-loop proxy | In-loop `val.oracle_ade@2s` = `flagship_v15.py:610`. step 10000 = **0.360 (> 0.30)**; oscillates 0.30–0.39. Held-out matched-anchor tool (`refc_scale_ab.py`) is REF-C-only, not v4. |
| `miss_at_2m ≤ 0.10` (KILL) | ⚠️ needs the missing windows dump | `driving.tier0`/`bench` emit it, but only from a v4 `windows` dump (blocked above). In-loop `val.miss@2m` step 10000 = **0.249 (≫ 0.10)**. |
| `speed_benefit_recovered_frac ≥ 0.70` (KILL) | ❌ **not v4-native** | Formula only in `postmortem_a/b_analyze.py` over train logs; trainer has a comment, no emitter. |
| `seam_norm_ratio_max ≤ 1.0` (KILL) | ❌ **not persisted** | Computed `flagship_v4.py:193` but dropped by the log whitelist `train_flagship_v4.py:634-635`; absent from `metrics.json`. (Clamped ≤1.0 by construction.) |
| `encoder_touching_levers ≤ 2` (KILL) | ➖ constant `2` | Operator supplies 2.0. |
| `deploy_tick_p99_ms ≤ 50` (KILL) | ⚠️ proxy only | `incoming/2026-07-22-orin-thor-deployment/.../bench_latency_report.json` is a **predictor-only A40 proxy**, explicitly "NOT the full tick" (p99 5.08 fp32 / 27.88 graph-k20 / 4.36 fp16). No measured full deploy tick. |
| `nonav_route_beats_majority ≥ 1` (KILL) | ❌ needs glue+rename | `hierarchy.py:423` computes `vision_route_beats_majority`, but `hierarchy.run` requires `model.strategic_policy` (`:229`) which the v4 ckpt's separate `GoalScalarHead` does **not** expose. |
| 5 report-only (`strat_subspace_*`, `imag_win_at_5s`, `longh_5s_beats_persistence`, `cruise_delta_vs_holdv0`) | ❌ 4 unbuilt, 1 differently-named | Confirms LOOP_STATE's "P7 unbuilt". |

`run_gate` **ANDs all 8 KILL** secondaries and returns `INCOMPLETE` if any is unsupplied → **even with a held-out eval JSON, the gate could not complete today** (≥3 KILL secondaries have no runnable v4 producer).

## MEASURED in-loop health at step 10000 (trainer diagnostics — **NOT** the gate primary)

From a **read-only** copy of `pod2:/workspace/experiments/flagship-v4.1-30k/train.log` (`v4.1_train.log`, staged; provenance `v41_step10000_inloop_health.json`). **This is not an eval-on-pod2** and is **not admissible as the gate primary** (wrong metric, wrong path, no CI; CLAUDE.md: *"only eval output is quotable"*).

| in-loop metric @ step 10000 | value | card bar (held-out, different metric) |
|---|---|---|
| `canary_ade@2s` (plan-free WM) | **0.45994**, `vs_base +0.038`, controller **"ok"** | wm_canary ≤ 0.55 |
| `val.ade@2s` (mean-over-anchors) | **0.7054** | primary ≤ 0.60 |
| `val.oracle_ade@2s` | **0.3598** | oracle_in_fan ≤ 0.30 |
| `val.miss@2m` | **0.2486** | miss_at_2m ≤ 0.10 |
| controller `lam_mult` | **1.53e-05** (planner grad ≈ OFF since ~step 2000) | — |

⚠️ **Two caveats that change the story from LOOP_STATE's "flat, healthy":**
1. **The canary is healthy partly *because* the planner is gradient-starved.** The λ_plan controller has been *halving* the planner gradient at nearly every eval since step ~2000 (`lam_mult` 2.4e-4 → … → 1.5e-5 by 10k). The canary at step 10000 is "ok" (0.460), but its neighbours breach: **0.603 @ 9000, 0.633 @ 11000**. The WM was protected by turning the planner off.
2. **Directional (HYPOTHESIS, not a verdict):** in-loop `val.ade@2s = 0.705 > 0.60` bar and > v1's 0.452; oracle 0.360 > 0.30; miss 0.249 ≫ 0.10. If the held-out produced-goal eval tracks these, the **primary and ≥2 KILL secondaries would plausibly FAIL** — i.e. v4.1 at 10k reads as "healthy WM, under-trained planner." **This must be confirmed by the real held-out eval, not asserted here.**

## Blockers to actually running the gate (each MEASURED)

1. **No v4 held-out eval driver** — new glue required: build `FlagshipV4Head(V4Config)`, load `ck['model'|'head'|'grounding'|'goal_head']`, run the **produced-goal** path over 40 val eps, save `pred/gt/cv/eid/speed/head_deg/wp_steps` → `driving.from_windows()` → `cluster_bootstrap`. (`eval_flagship_v16.py` is the template; its `collect()` body is duck-compatible but `main()` is v15-hardwired.)
2. **≥3 KILL secondaries have no v4 producer**: `speed_benefit_recovered_frac`, `seam_norm_ratio_max` (persist it), `nonav_route_beats_majority` (wire the v4 produced-goal path); `deploy_tick_p99_ms` needs a real full-tick number; `oracle_in_fan`/`miss_at_2m` need the windows dump.
3. **pod1 unprovisioned (MEASURED):** val cache `physicalai-val-0c5f7dac3b11` **absent** (only train cache present); **taniteval package absent**; **`flagship_v4.py`/v15 code absent** (pod1 `stack` is Jul-11). torch 2.4.1+cu124 OK, GPU idle, disk 1.0 GB/s OK.
4. **ckpt (3.24 GB) still on pod2.** HF fast-relay is **likely blocked** — the v16 push already hit `403 private storage limit reached` (LOOP_STATE); a 3.24 GB ckpt would too. Fallback dev-box relay is ~1 MB/s (~2×53 min). Needs a transfer decision.
5. **run_gate ↔ v4-trainer log-schema mismatch:** the card's `compare_metric g_op_fwd_ade_m` is **not logged** by the v4 trainer (it logs `oracle_ade`/`plan_ade`) and it logs `elapsed_s`, **not** `step_s` → the gate's comparative-ratio-vs-v1 and GPU-hour/budget diagnostics **crash / read 0/NaN** on a v4 log (verified: `check` `REFUSING: no matched steps`; budget `0.0 GPU-h | nan s/step`). The harness was only ever validated on **v1's** log/windows (`v1_g1_dryrun_gate_FIXED.json`), never a v4 artifact.

## What must happen next (escalation payload, ranked)

1. **Decision for Sayed:** the v4.1 10k gate is a **build task**, not a run task (~a v4 eval driver + 3–4 KILL probes + pod1 provisioning + a 3.24 GB transfer). Approve building it, or accept the in-loop directional read above as the interim signal. Given caveat (1)+(2), the cheaper discriminating question may be *"is the λ_plan controller starving the planner?"* rather than a full gate.
2. **Build & validate the v4 eval driver** (`eval_flagship_v4.py`) against a known ckpt first (reproduce a published number) before it is allowed to gate — O-03 discipline.
3. **Provision pod1**: scp `taniteval/` + updated `stack/` (with `flagship_v4.py`, `flagship_v15.py`, v15 aux) + the val cache (or rebuild val from HF), then move the ckpt.
4. **Reconcile the log schema** (or patch run_gate to read `oracle_ade`/`elapsed_s`) so the comparative + budget diagnostics compute on a v4 log.
5. **Restart-budget note:** the card carries `restarts_used 0` (verbatim pre-registration). v4.1 IS the lr_trunk restart of v4 within lever-family `joint-planner-wm`; whoever runs the real gate should decide whether to bump `restarts_used` to 1 (affects only the FAIL branch: `RESTART` vs `REFUTE_LEVER_FAMILY`).

## Deliverable manifest

| artifact | where | one place only? |
|---|---|---|
| this status/escalation | `repo:…/incoming/2026-07-23-v41-10k-gate/STATUS_BLOCKED.md` (STAGED) | no |
| **materialized pre-registered card** | `repo:Project Steering/Gates/flagship-v4.card.json` (STAGED) + copy in incoming | no |
| gate result (verdict **BLOCKED**) | `repo:…/incoming/2026-07-23-v41-10k-gate/v41_10k_gate_result_BLOCKED.json` (STAGED) | no |
| v4.1 train.log (read-only copy, 257 rows) | `repo:…/v4.1_train.log` + `pod2:/workspace/experiments/flagship-v4.1-30k/train.log` | no |
| v4.1 config.json | `repo:…/v4.1_config.json` + pod2 | no |
| in-loop health @10k (structured) | `repo:…/v41_step10000_inloop_health.json` (STAGED) | no |
| **ckpt_step10000.pt (3.24 GB)** | **`pod2:/workspace/experiments/flagship-v4.1-30k/ckpt_step10000.pt` ONLY** | ⚠️ **YES — single pod disk** |

⚠️ The 10k gate checkpoint lives on **one pod disk only**. It is not at risk of recycle (v4.1 still training into the same dir), but it should be HF-backed once a transfer path is chosen.
