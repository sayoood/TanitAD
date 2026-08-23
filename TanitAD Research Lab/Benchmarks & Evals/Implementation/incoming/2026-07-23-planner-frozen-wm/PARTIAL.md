# planner_on_frozen_wm — PARTIAL (superseded, stood down at step 300)

**2026-07-23 (Berlin UTC+2).** Coordinator stood this experiment down mid-run: Sayed decided to go
straight to **v4.2 (joint, no frozen part)**, so the frozen-WM discriminator is superseded and pod3 was
released for own-encoder Branch B. The run was **KILLED at step 300 of a 10,000-step read** — far before
the pre-registered gate — so there is **NO held-out MODE B `ade_0_2s`** and the STARVED-vs-BAD-BY-DESIGN
verdict is **UNRESOLVED**.

**Interim numbers banked (both MEASURED, evidence class noted):**
- **Setup validated (decision-grade):** MODE A v1-canary on the rebuilt disjoint held-out val =
  **0.42535** (Δ −0.0018 vs v1's 0.4271, `HARNESS_VALIDATED: true`; `v1-canary-heldout-pod3.json`). The
  frozen trunk's **step-0 canary = 0.42535** confirmed the WM was frozen at v1's known-healthy level;
  `lr_trunk` = 0.0, `lam_mult` = 1.0 throughout.
- **Planner learning signal (C1 — in-loop TRAINER log, dense-20, PRE-CONVERGENCE, NOT the gate metric,
  NOT quotable as a result):** in-loop `plan_ade` fell **69.7 → 62.4 → 30.5 → 17.6 → 7.03 → 4.37** over
  steps 0/50/100/150/250/300 — the planner head was training normally on the frozen WM and descending
  fast, but step 300 is ~3% of the read and the dense-20 in-loop metric is not comparable to the 4wp
  held-out `ade_0_2s` (0.4271 v1 / 0.8522 v4.1). It says only "the head was learning," nothing about the
  STARVED-vs-BAD verdict.

**Reusable if this question ever reopens (all staged/on pod3, provisioning was the expensive part):** the
disjoint held-out val (`valcache/physicalai-val-heldout-79d4e3d2d4c6`, 44 eps, O-03 validated), the
`[256,20,2]` dense anchors, the v1 trunk, the deployed v4 code, and the exact frozen-WM launch command
(`frozenwm_run_config.json`, = v4.1 canonical + `--lr-trunk 0 --lambda-plan 1`) all lived under
`pod3:/workspace/v4run/` — but pod3 is being repurposed, so treat that as gone. See `STATUS.md` for the
full provisioning record.
