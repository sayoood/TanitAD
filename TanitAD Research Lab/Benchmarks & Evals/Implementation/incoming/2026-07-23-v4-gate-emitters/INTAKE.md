# v4 gate — the 3 missing secondary emitters (KILL secondaries)

**Date:** 2026-07-23 (Berlin) · **Author:** gate-emitters subagent · **Status:** built + validated, STAGED (not committed)

**Mission:** the v4 gate card defines three KILL secondaries with **no emitter**, so `run_gate.py`
returns **INCOMPLETE** on any v4 checkpoint even with a decisive primary (this made v4.1's 10 k gate
formally INCOMPLETE). Build + validate the three emitters against the deployed flagship **v1** ckpt as
the test fixture, wire them so `run_gate.py` reads them, and confirm the gate then renders **COMPLETE**.

---

## STEP 0 — the KILL-vs-report reconciliation (banked)

**Discrepancy:** `LOOP_STATE.md` framed these three as "P7, report-only, NON-blocking"; the Registry
agent read the card as "**KILL** secondaries."

**Verdict: the Registry agent is correct. All three are KILL.** Per-emitter, read off
`Project Steering/Gates/flagship-v4.card.json` directly:

| secondary | on the card's `secondary` array? | KILL or REPORT |
|---|---|---|
| `speed_benefit_recovered_frac>=0.70` | **yes** | **KILL** |
| `deploy_tick_p99_ms<=50` | **yes** | **KILL** |
| `nonav_route_beats_majority>=1` | **yes** | **KILL** |

Two independent confirmations:

1. **Mechanism (`run_gate.py`).** `cmd_check` iterates `card.secondary`; any card secondary with no
   supplied `--secondary-value` is written `pass: None`, and `any(pass is None)` → verdict
   **INCOMPLETE** (`run_gate.py:578-619`). There is **no report-only flag for card secondaries** —
   report-only is a *separate off-card channel* (`--secondary-value` keys NOT matched to a card
   secondary; `run_gate.py:607-615`). So **every on-card secondary is KILL by construction.**
2. **Design (`V4_FLAGSHIP_DESIGN.md` §9 split-card table).** Explicitly marks all three **KILL**
   ("speed_benefit… the quiet plateau [PM] found in v3enc"; "deploy_tick… the arm is undeployable";
   "nonav_route… the one hierarchy read that is genuinely pass/fail").

**Root of the discrepancy (C4/conflation):** LOOP_STATE's "P7 report-only" list is a **different,
off-card set of five falsifiers** — `imag_win_at_5s`, `strat_subspace_{sufficiency,compression}`,
`longh_5s_beats_persistence`, `cruise_delta_vs_holdv0` — which ARE genuinely report-only (§9). The two
sets were merged in prose. **Build order followed the KILL priority: all three block a complete verdict.**

---

## The three emitters (built + validated on flagship v1, `flagship-30k`, step 29999)

### 1. `deploy_tick_p99_ms` ≤ 50 — **KILL** — PASS on v1 (18.76 ms)
- **Definition:** p99 (not mean) of the **fully-composed deployed inference tick** — encode + rollout
  with every accuracy-preserving lever on (fp16 weights + rolling encoder-state cache + whole-rollout
  CUDA graph = the `all_levers` lever), over 200 warmed ticks.
- **Emitter (§17.3):** `taniteval.efficiency` lever panel. `gate_emitters.deploy_tick_from_eff_json`
  reads the panel's `eff_levers_<key>.json`, selects the most-composed **accuracy-equivalent** lever
  (guard: `|ade_0_2s_delta_m| ≤ 0.05`, rejects a fast-but-wrong tick), and surfaces its `p99_ms`.
- **v1 value (MEASURED, A40, `taniteval/results/eff_levers_flagship-30k.json`):** deployed lever
  `all_levers`, **p99 = 18.7641 ms** (mean 18.7631, std 0.19), accuracy delta −6.6e-05 m
  (cosine 0.99999994), GPU-exclusive → **PASS (≤ 50).**
- **Fresh on-pod confirmation THIS session (MEASURED, pod1 A6000, reusing `efficiency.build_levers`,
  `artifacts/eff_levers_flagship-30k_LIVE.json`):** `all_levers` **p99 = 16.8858 ms** (std ~0.2), accuracy
  delta +0.0011 m → the emitter reads it end-to-end → **PASS.** ⚠️ flagged non-exclusive (a resident
  3.8 GB sidecar at **0 % util** — no contention; timings tight); the A40 artifact is the exclusive
  reference. Two GPUs bracket the value at **~17–19 ms**, both ≪ 50.
- **Orin/TRT cross-check (mission):** matches the "composed tick ~18.75 ms mean" exactly. The Orin
  artifact's predictor-only graph-step rollout p99 = 27.884 ms (fp32) is the rollout sub-number.
- **v4 delta (a knob, per §8):** v4's operative predictor is v1-verbatim; the anchored-diffusion head
  adds `diffusion_steps` truncated-denoise passes → ~25–28 ms floor with the imagination probe, still
  ≪ 50. For a v4 arm, run the efficiency lever panel on the v4 ckpt (the head is timed in the tick),
  then point the emitter at that `eff_levers_<v4key>.json`.

### 2. `speed_benefit_recovered_frac` ≥ 0.70 — **KILL** — PASS on v1 (0.8184)
- **Definition (pinned, `postmortem_a_analyze.py:205`, do NOT re-derive):**
  `(nospeed_control − arm) / nospeed_control` on the **bucket-mean `g_op_fwd_ade_m`** over the gate
  bucket `(8000, 10000]` (convention `lo < step ≤ hi`). 1.0 = zero operative-rollout error (full speed
  benefit); 0.0 = no better than a trunk with no speed channel. Catches the **quiet v3enc plateau** the
  WM canary misses.
- **Emitter (NEW):** `stack/tanitad/eval/speed_benefit.py` (the design's P8 target). Zero-GPU — reads the
  arm's `train_log.jsonl` + the git-tracked no-speed control log.
- **v1 value (MEASURED, `taniteval/results/trainlogs/`):** **0.8184** → **PASS (≥ 0.70).** Reproduces the
  design's headline **81.8 %** exactly; v3enc reproduces **0.1859 = 18.6 %** exactly (both pinned in a test).

### 3. `nonav_route_beats_majority` ≥ 1 — **KILL** — FAIL on v1 (0, correctly)
- **Definition (§7A.5):** with the nav command **withheld** (follow), does the route head's accuracy on
  the valid subset beat the majority-class (always-straight) base rate? The strategic-route-value
  falsifier: an arm whose *produced* goal cannot beat "always straight" has relabelled a command echo,
  not built a strategic level.
- **Emitter (§17.3):** `taniteval.hierarchy` — the `vision_route_beats_majority` JSON key (under
  `seam_nav_to_strategic`, vs `majority_straight_rate`). `gate_emitters.nonav_route_from_hierarchy_json`
  surfaces it as int 0/1. **Flagship path is clean** — this uses the follow-command hierarchy, NOT the
  REF-C `nav_cmd=None` confound (RETRACTION_LOG 07-21).
- **v1 value (MEASURED THIS SESSION, pod1, `artifacts/hierarchy_flagship-30k_v1.json`):**
  `route_acc_follow` **0.7083** == `majority_straight_rate` **0.7083** exactly (the follow head predicts
  **straight on all 72/72** valid windows) → `vision_route_beats_majority` **False** → **0** → **FAIL**.
  This is a clean MEASURED confirmation of v1's pure **command echo** (`route_skill_vs_chance = 0.0`;
  commanded `route_acc_nav` = 1.0 by construction, zeroed-nav = 0.236). The "0.861" in RETRACTION_LOG is
  the encoder *speed* probe R², not route — checked against MODEL_REGISTRY.md:376.

---

## The v1 dry-run — the gate now renders a COMPLETE verdict (§17.1b)

Throwaway card `artifacts/flagship-v4-dryrun.card.json` (byte-identical secondary set to
`flagship-v4.card.json`); primary from the committed `driving_flagship-30k.json` (`ade_0_2s` 0.4271,
episode-cluster bootstrap CI [0.3675, 0.4871] → PASS ≤ 0.60).

| run | secondaries supplied | verdict |
|---|---|---|
| **A** | 5 existing only (omit the 3) | **INCOMPLETE** — `deploy_tick_p99_ms`, `speed_benefit_recovered_frac`, `nonav_route_beats_majority` NOT SUPPLIED (the status quo that made v4.1 INCOMPLETE) |
| **B** | **all 8** (5 existing + the 3 emitters' MEASURED values) | **RESTART** — a **COMPLETE** verdict; all 8 adjudicated |

**The 3 emitters flip INCOMPLETE → COMPLETE.** Run B's verdict is RESTART (not CONTINUE) because two
KILL secondaries genuinely fail on v1: `oracle_in_fan` 0.3073 > 0.30 (the frozen-fan floor v4 must beat)
and `nonav_route_beats_majority` 0 < 1 (v1's command echo). Both are v1's *known* properties — the
machinery renders a full, honest verdict. A converged v4 arm passing those two gets CONTINUE (pinned in
`test_with_the_three_emitters_the_gate_renders_a_COMPLETE_verdict`). Artifacts:
`v1_dryrun_gate_A_incomplete.json`, `v1_dryrun_gate_B_complete.json`.

---

## Findings banked (per the operating standard)

- **⚠️ The v4 trainer did not log `g_op_fwd_ade_m`, so a real v4 arm's log was NOT gate-computable for
  `speed_benefit_recovered_frac`.** It is *computed* (`train_flagship_v4.loss_step`, in the `log` dict)
  but was **dropped from the written row** (only `total/lambda_plan/wm/planner/plan_ade/oracle_ade` were
  kept). The reference arms (flagship4b trainer) log it; the v4 trainer logs `canary_ade@2s` instead
  (a **different quantity/scale**: v1 canary 0.452 vs g_op_fwd 0.105 — not interchangeable). **Fix
  applied:** added `g_op_fwd_ade_m` to the logged-row key tuple (`train_flagship_v4.py`, LOG-ONLY, `if k
  in log`-guarded, no loss/parity effect). Existing v4.1/v4.2 logs still lack it → their
  `speed_benefit_recovered_frac` reads NOT SUPPLIED (the emitter refuses to fabricate a pass); future v4
  launches are gate-computable.
- **v1's route head is a pure command echo — MEASURED, not inherited** (72/72 straight under follow;
  route_acc_follow == majority exactly). Confirms the design's `route_skill_vs_chance = 0.0`.
- **deploy_tick cross-check reconciled:** committed `all_levers` composed tick p99 = 18.76 ms = the
  "~18.75 ms" cited in LOOP_STATE/deployment; the Orin 27.884 ms is the predictor-only rollout p99.

---

## Deliverable manifest (all STAGED in the working tree, NOT committed)

| artifact | path (repo, absolute-in-tree) |
|---|---|
| speed_benefit emitter (P8 target) | `stack/tanitad/eval/speed_benefit.py` |
| 3-emitter orchestrator + `gate-values` | `stack/scripts/gate_emitters.py` |
| tests (17, green; full suite 803 passed / 2 skipped) | `stack/tests/test_gate_emitters.py` |
| v4-trainer log-key fix (log-only) | `stack/scripts/train_flagship_v4.py` (1 line) |
| throwaway dry-run card | `…/incoming/2026-07-23-v4-gate-emitters/artifacts/flagship-v4-dryrun.card.json` |
| v1 gate secondary values (emitted) | `…/artifacts/v1_gate_secondary_values.json` |
| dry-run A (INCOMPLETE) / B (COMPLETE) | `…/artifacts/v1_dryrun_gate_{A_incomplete,B_complete}.json` |
| v1 hierarchy result (MEASURED, pod1) | `…/artifacts/hierarchy_flagship-30k_v1.json` |
| v1 efficiency lever panel (committed copy) | `…/artifacts/eff_levers_flagship-30k.json` |

**Pod state:** measured on `tanitad-pod` (pod1) under `gpu_lock` owner `gate-emitters`; lock released,
pod idle. **Escalation:** the `train_flagship_v4.py` log-key fix should ride the next v4.x launch so its
10 k gate is fully computable — flagged, not left in a doc.

## How a real v4 arm's 10 k gate is run (workflow)

```
# 1. efficiency lever panel on the v4 ckpt  ->  eff_levers_<v4>.json  (includes the diffusion head tick)
# 2. hierarchy panel on the v4 ckpt          ->  hierarchy_<v4>.json
# 3. assemble the three secondary values:
python stack/scripts/gate_emitters.py gate-values \
    --eff-json  results/eff_levers_<v4>.json \
    --hierarchy-json results/hierarchy_<v4>.json \
    --arm-log   results/trainlogs/<v4>_train_log.jsonl
# 4. paste its --secondary-value line (plus the 5 existing) into run_gate.py check.
```
