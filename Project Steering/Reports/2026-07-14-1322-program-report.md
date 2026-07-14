# TanitAD — Program Report · 2026-07-14 13:22 UTC

**Headline:** Flagship at **10.6k/30k**; its **10k gate = 3.54 m** grounded ADE — now **below fully-trained REF-A (3.73 m) at only a third of training**. REF-A's weakness **localized to speed/scale** (not rotation/imagination), and the first fix (**v1 aux ego-motion supervision**) shows a strong early signal (latent speed-R² **−2.14 → 0.82**). All three arms healthy.

> Scope note: the hub `PROJECT_STATE.md` was last updated 2026-07-11 (Orchestrator W31) and describes the *pre-4-brain realmix* era; this report covers the **current flagship4b 3-arm parity run** + the REF-A diagnosis/improvement track, which are ahead of it. The hub state needs an orchestrator refresh (item 8).

---

## 1. Main training — flagship4b-phase0-30k (pod2, A40)

| Field | Value |
|---|---|
| Step | **10,600 / 30,000 (35.3%)** |
| Measured pace | **~9.0 s/step** (`step_s` 449.3 / 50-step interval; ~400 steps/h) · data_s 0.65 s/step (not data-bound) |
| Loss | 3.04 (sigreg 12.4, ground 0.914, inv 0.236) |
| Health | erank 17.3, dim_std 0.807 → **no collapse**; man_acc 0.44, route_acc 1.0 |
| Watchdog | heartbeats live (`flagship-phase0-30k.json`, `ops-daemon.json`); disk `tight:false`, dd-500 MB OK, 2376tr+600va; `oom_kill_total:38` (accumulated, non-fatal — ops-daemon drops page-cache) |
| ETA | next gate (15k) **~11 h** · 30k **~2.0 days** |

**In-training per-brain grounding (meters):** operative fwd ADE **0.644** / mid de 1.42 · tactical fwd 2.22 / mid 5.41 · strategic fwd 2.52 / mid 4.05. Operative brain is strongest; tactical/strategic still maturing.

*Pace caveat (honest):* the 3-min wall-clock probe read flagship Δ0 (10600→10600) — this is the known **50-step log-granularity artifact** (log updates every ~450 s), NOT a stall. `step_s` is the authoritative per-step measure.

---

## 2. Bake-off arms (3-arm parity comparison + REF-A improvement variants)

| Arm | Pod | Step | Pace | Key numbers |
|---|---|---|---|---|
| **Flagship** (4-brain) | pod2 | 10,600/30k | ~9.0 s/step | see §1; man_acc 0.44 |
| **REF-A** (frozen-DINO) | pod3 | **done 30k** | — | grounded ADE **3.726 m** (< CV 0.825 = FAIL); fully diagnosed |
| **REF-B** (from-scratch ViT) | pod1 | **2,800/30k (9.3%)** | ~24.1 s/step (data-bound) | loss 99.4 (679→99); man_acc 0.64, route_acc 1.0, nav_follow 0.94; ood_score 1.77; gnorm 282 (elevated, non-diverging); ETA ~7.6 d |
| **REF-A v1** (+aux ego-motion) | pod3 | training | ~diagnostic 12k | early mechanism: aux_speed_r2 **−2.14→0.82**, fwd_ade 3.35→0.83 (@step 300); held-out 12k gate imminent (agent-driven) |

Same-data parity intact: all arms on val `physicalai-val-0c5f7dac3b11` (byte-identical); train cache key `e438721ae894`, skip-hash `f09e44db`.

---

## 3. Experiments / evals completed since last report

- **Flagship gates:** 5k → 4.31 m, **10k → 3.54 m** (held-out ade_0_2s ±0.35). Trend **7.18 → 4.31 → 3.54** — crossing under REF-A's fully-trained 3.73 m at a third of training.
- **REF-A full benchmark suite:** D1 **FAIL** 7.65 m · D2 **PASS** 0.87/0.89 · D3 **BLOCKED** (I4 6.4) · behavior probes above-chance-below-majority · D8 in-domain baseline only (real OOD-separation blocked — no degraded cache).
- **REF-A 4-test ablation diagnosis:** root = **speed/scale magnitude (71–83%)** + frozen-encoder ceiling; **imagination exonerated (11%)**; shape/rotation already good (0.31 m). *(Memory: [[refa-bottleneck-diagnosis]].)*
- **1a post-hoc calibration:** negative — no global bias, per-window scale not decodable from frozen latent (scale-probe R² −2.19) → **retrain required**; oracle headroom 68% (3.69→0.92 m).
- **REF-A fixes 1b/2/4** implemented + smoke-tested (isolated on pod3).
- **REF-A visualizations:** 8 BEV + 8 real-camera-projected overlays (real f-theta calibration), published as a phone gallery.

---

## 4. Agent updates (git + hub)

- **Recent commits:** `97edaaf` phase0-eval behavior gate + turnkey watch_gates; `827805d` TanitResim eval unification; `0c6efe8` 3-arm gate-comparison harness.
- **Uncommitted hub-agent work (pending orchestrator sweep):** `PROJECT_STATE.md`, `Benchmarks & Eval/BACKLOG.md`, `Research/STATE.md` modified; new intake packages `2026-07-13-cosmos-robustness-first-pass`, `2026-07-13-backlog3-synthetic-corpora`; untracked `OWN_DATASET_PLAN.md`, `eval_metric_rollout.py`, `validate_data.py`, `test_physicalai_rig.py`.
- **Backlog movement:** eval harness matured (behavior gate + TanitResim + 3-arm compare_arms) but is **NOT deployed on the pods** (they're detached at `0f93b98`); per-arm grounded evals were ported ad-hoc (flagship native; REF-A `/workspace/tmp/refa_eval/`). This is the gap behind the blocked D4–D9 gates.

---

## 5. Program position — Master Plan phases & the four edges

- **Phase 0** (foundation & first edge proofs), ~day **10/42**; final evaluation (P7) **2026-10-05**. The core Phase-0 proof — *does the 4-brain beat the flat/frozen references on identical data?* — is **in progress and trending yes** (flagship leads REF-A at ⅓ training).

**The four edges:**
| Edge | Status |
|---|---|
| **Hierarchy** (4-brain) | Operative strong (fwd ADE 0.64 m in-training); tactical/strategic maturing (2.2/2.5 m). Verdict at flagship@30k vs REF-A/REF-B. |
| **Efficiency** (CNCE) | 261 M vs 15–32 B competitors; 15 ms/tick baseline. On track (no regression this cycle). |
| **Imagination** (H15) | REF-A imagination D2-passes (directionally usable). **Watch:** flagship log shows `h15=0.0` — verify the imagination loss is active in flagship4b (item 8). |
| **Self-monitoring** (OOD/epistemic) | REF-B carries `ood_score` (currently frozen); the real clean-vs-degraded **D8 separation is blocked** (needs a degraded feature cache). |

---

## 6. Next steps (ordered)

1. **v1 held-out 12k gate** (imminent, agent-driven) → confirm the speed-fix works on val → then **v2** (+temporal adapter), **v3** (+rollout-k).
2. **Flagship 15k gate** (~11 h).
3. **Flagship → 30k** (~2 days) → decision-grade flagship-vs-REF-A verdict + full benchmark suite on the flagship.
4. **REF-B → 30k** (~7.6 days) → completes the 3-arm comparison (or sooner if the speedup is approved).
5. Deploy the eval harness + degraded cache to the pods → unblock D4–D6 / D8-separation / D9 (needs Sayed's go).

---

## 7. Decisions required from Sayed (with defaults)

| # | Decision | Default if quiet |
|---|---|---|
| A | **REF-B speedup** (workers failed on shm; alt = grad-ckpt-drop ~6.7 d, or a pod1 memory-reconfigure ~4.5 d) | Leave at ~7.6 d (reference arm) |
| B | **V-JEPA / motion-encoder REF-A-v4** | **Hold** — v1's strong signal says the cheap fix has headroom; revisit if v1–v3 plateau |
| C | **Deploy eval-harness + degraded-cache to pods** (unblocks D4–D6/D8-sep/D9) | Hold until flagship nears 30k |
| D | **Scorecard panel in the gallery** | Hold |
| E | Hub §4 carryovers: D-021 (keep 2048), D-022 firewall (legal), MetaDrive source-install (~10 min Sayed task) | Unchanged |

---

## 8. Improvements / incidents (honest)

- **Monitor fixed:** REF-B's log had a stray binary byte from the earlier workers-crash → plain grep bailed → "?" blind-spot. Root-caused + fixed with `grep -a`; rebuilt as a clean 2-arm monitor (`bqaj2abzy`), dropped the done REF-A (was false-`STALL`-spamming), added a `[x]`-trick pgrep so a real death registers.
- **REF-B workers=6 speedup FAILED** — `/dev/shm` SIGBUS (6 workers page-faulting the 260 GB mmap under pod1's 62 GB cgroup); the `file_system` bypass also failed. Reverted safely to `num_workers=0` (~200 steps lost).
- **1a calibration negative** — informative: ruled out the no-retrain shortcut, confirmed the retrain path + a 68% ceiling.
- **10k-gate background task → harness zombie** (result captured in `gate_step10k.json`; task stopped). The "lingering eval process" I flagged was my own command **self-matching** the search string — caught via `ps`/`etime` before acting (no false kill).
- **pod2 `oom_kill_total=38`** (accumulated, non-fatal; ops-daemon manages page-cache; run healthy).
- **WATCH — `h15=0.0`** in the flagship log: verify the H15 imagination loss is active/weighted in flagship4b.
- **Hub `PROJECT_STATE.md` ~3 days stale** (07-11 W31, realmix era) — needs an orchestrator refresh to the flagship4b 3-arm reality.

---
*Fresh measurements taken 2026-07-14 13:15–13:22 UTC (pod tails + 3-min pace probe). Grounded gate numbers are held-out 8-split means. Next report on the D-025 cadence or on the v1 verdict, whichever lands first.*
