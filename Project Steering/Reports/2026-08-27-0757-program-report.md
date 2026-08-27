# TanitAD program report — 2026-08-27 morning (D-025)

**Author** Master Mind · **Window** 2026-08-26 19:00 → 2026-08-27 08:45 (Europe/
Berlin; pods/Thor log UTC) · **Branch** `agent/arch-inf-20260803`, 14 commits this
window (`7c22ca7`…`ed62e16`) · Every number: arm + tier + evidence class; sources
are the registry and raw eval JSON only.

---

## 1. Fleet, measured this hour

| box | state |
|---|---|
| **Thor** | ▶ `k4_30k` (A7, the fourth 2×2 corner) step 800, `step_s` 0.923 — ETA ~16:30. Overnight it completed: the D1 crossed cell (30k), the first parity T1 (both arms), D2 `omega30k` (30k) |
| **dev box** | idle; overnight ran E-DEC-63 (80 clips), F1–F4, the D1 drift/nrmse reads, the D2 reads + seed-noise bound |

## 2. The night's results (each committed the turn it landed)

| result | verdict | evidence class / tier |
|---|---|---|
| **E-DEC-63 residual ceiling** | ceiling beyond drift is **small (~+0.01 vs drift +0.6722) but real and PIXEL-BORNE** (+0.0096, t 5.11); our predictor at the ceiling (−0.0023) | MEASURED, T0, arm `rdw8p30k`, rig valid (A1 = banked value to 4 decimals) |
| **F1–F4** | NOT photometric (lum3 null) · NOT probe dilution (pixels+noise t 9.10) · **the token field actively displaces it** (tokens-alone t −3.95) | MEASURED, T0 |
| **D1 encoder policy** | **DEGENERATE** per prereg: drift 0.3905 ✅ but nrmse 0.9301 vs 0.8115 (+14.6% ⛔) ⇒ **encoder TRAINABLE** | MEASURED, T0, pre-registered bands |
| **First parity T1** (programme first) | **neither arm drives** — ADE 14.52/14.21 m vs CV floor 0.5352; `_longitudinal_claim_admissible: False` both; speed bias ≈ −mean speed = **the drift predictor's T1 face**; S-contrast 0.2632 vs 0.0175 separated but oscillation-vs-inertia | MEASURED, **T1**, parity corpus, episode-cluster bootstrap |
| **D2 conditioning** (`[ω,a,v]`, PI channel) | **MIXED as committed:** marginal null (t −1.29) — E-DEC-48b extends; drift unchanged; ⭐ **nrmse improved 14.8%** (0.8115→0.6911), ~4× the freshly-measured 3.5% seed band — **`omega_accel_v` ADOPTED for v7r**, replicate queued before un-hedged quoting | MEASURED, T0, pre-registered |
| **A7 redefinition** | config diff proved `o5_k` is the ONLY substantive `splitp30k` knob ⇒ the informative cell is trainable+k4 (running); additive ≈0.48 / interaction ≈0.67 committed | MEASURED diff; arm ▶ |
| **JEPA-driving survey** (PI-requested) | Drive-JEPA 93.3 PDMS @307M/208h · Latent-WAM 89.3 EPDMS · latent-subgoal hierarchy 70% vs 0% flat — **the combined position (hierarchy × driving × latent WM) is unoccupied** | PUBLISHED, 5 primaries banked (Library 74–78) |

## 3. Programme position vs the L-ladder (V7_RECIPE_AND_SCALEUP §8)

L0–L2 ✅ instruments + representation proven · **L3 measured-and-open** (no arm's
predictor beats its own `z_t` on content; E-DEC-63 says the current latent's
ceiling is the blocker, not predictor capacity) · **L4 measured-and-failed at
parity** — the honest baseline, with the harness (incl. anti-echo) proven
end-to-end · L5 awaits REF-D + the hierarchy-traversing eval.

## 4. Design state

- **v7r proposal committed** (`V7R_DESIGN_PROPOSAL.md`, PI-corrected §1.5): four
  brains kept; latent-subgoal interfaces; diffusion planner in the
  **(a,κ)-unicycle action space** (feasible-by-construction; free-XY refuted 25×/
  97.6% infeasible); v1.7 decoder learnings carried; O14-fut (R2) **approved**;
  occupancy target adopted; `omega_accel_v` adopted.
- **PREREG_O14** committed (E-DEC-67): 5 tiny arms incl. time-shuffled deliberate
  regression; primary read = the E-DEC-63 absorption metric.

## 5. Ordered next steps

1. **Implement O14-fut/O14-rec behind flags** (default bit-identical, test-pinned) — today.
2. **A7 read** (~16:30): drift vs the three committed bands → closes the 2×2.
3. **O14 tiny ladder** on Thor after A7 (~2 h) → gates decide the 30k arm.
4. **Bank D2/A7 raws** into an incoming package.
5. **Stage B commissioning** (obstacle.offline staging; tactical/strategic labels) — the critical path to L5.
6. omega seed replicate (low priority, before the 14.8% is quoted un-hedged).

## 6. Decisions for Sayed (defaults stated)

| decision | default if no direction |
|---|---|
| Fallback teacher: DINOv3 vs V-JEPA 2.1 | measure both (one probe column each) |
| NAVSIM(-style) external yardstick for Stage D | flagged only; no provisioning assumed |
| P0 bake-off budget (two-term vs EMA-teacher) | 4 tiny arms ≈ 2 h Thor |
| Stage B commissioning (Data FlyWheel) | waits for explicit go — it spawns agents |

## 7. Incidents, honestly

- T1 chain died twice before running (PYTHONPATH; then a **stale 08-03 taniteval
  package on Thor that predated the anti-echo gate** — shipped fresh, content-verified).
- The ps-grep launch gate **self-matched through its enclosing `bash -lc`** (third
  costume of the self-match family) — two-call pattern now standard.
- A stale `scoped-commit-index.lock` (47 min, no holder) blocked one commit — debris, removed.
- C166 logged: I quoted the wrong arm to the PI on "is representation solved" — the
  cross-arm rule (arm named beside every number) is now in force in every table above.
