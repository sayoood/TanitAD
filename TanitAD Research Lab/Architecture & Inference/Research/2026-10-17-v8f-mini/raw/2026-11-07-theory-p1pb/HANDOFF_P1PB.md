# HANDOFF_P1PB (P1′b agent, 2026-11-07). Folder `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/`. DONE; nothing staged, committed or pushed.

Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I` (never bare `python`), `CUDA_VISIBLE_DEVICES=-1`. Run times: control 4.5 min, holdout forward 4.5 min, read 3 min.

## Verdict (holdout alone, as THEORY_CRAWL_AMEND_2 and AMEND_1 §4 are written): **P1′b FAILS**
* Evaluable: 291 windows from 23 episodes; CONFIDENT 42 (11 episodes), UNSURE 125 (21 episodes), both seeds, floor 30 met.
* Δ median ρ(CONFIDENT) − median ρ(UNSURE): s0 +0.245 [−0.028, +0.407] NOT separated; s1 +0.312 [+0.068, +0.463] separated. The rule fails P1′ when Δ is not separated at either seed. Written consequence: T1′ is wrong too.
* Beside, deciding nothing: pooled eval-141 ∪ holdout HOLDS (66 / 200, Δ +0.301 [+0.140, +0.434] / +0.329 [+0.169, +0.510]). eval-141 alone (seen): +0.365 / +0.437.
* Sensitivities: Δ positive at both seeds in all six; G4 (eval grid stride) NOT EVALUABLE (12 CONFIDENT); `plan_raw` HOLDS (s0 lower bound +0.002); S1, S3, S4, S5 FAIL.

## Order and controls (all pass)
Admissibility `P1PB_ADMISSIBILITY.json` (A1 to A5, all pass) → `SPEC_P1PB.md` sha `f2cc7efb…` stamped 2026-11-07T00:07:38Z → control FW (driver reproduces the dumps' tick-0 plans, max abs 7.7e-05 m, 240 windows) → holdout forward (4,076 windows) → read. Also C4, F1, F2, N1 (1 / 0 of 20), N2, N3, probe artifacts unchanged. Probe AUC on holdout 0.748 [0.628, 0.849] (v0 3–6): not at chance.

## Judgement calls (fixed in the spec before scoring)
Primary = every window (stride 1), as AMEND_2 says; the eval grid (t = 3 mod 4) is sensitivity G4. 4 holdout clips have no row in the dev-box v9 release (f63ece41, not the 83e30bd3 C0 trained with): outside the primary set, inside S5. The 83e30bd3 vs f63ece41 row identity is INHERITED (`SPEC_REFCV8_DRAFT.md` line 1542).

## Next (the Master Mind's call; none run)
1. Register the verdict; the holdout windows are now seen too. Any further T1′ test needs FRESH windows: clips outside the 4,572 encoded → blocked on encode compute (box 2 or a pod) and a PI go. No unseen cached features remain on the dev box.
2. A fresh pre-registration on a continuous statistic (ρ against π over all windows, 124 of 291 fall in neither bin) with a committed power statement. Post hoc: median ρ sits 0.03–0.16 below the groups' median posteriors; CONFIDENT windows are slower and accelerate harder (v0 4.33 vs 5.51 m/s), a confound.
3. P4 on the LoRA arm stays the pre-registered next test of what never depended on T1′.

## Integration done here
`WSP_READ.md` §5 gained 5.1 and 5.2 (text added, nothing else changed; the §1 table row and title line still say NOT EVALUATED by instruction). `WSP_READ.json` gained keys `P1_prime_eval141_read`, `P1_prime_b_holdout_read`. If `make_wsp_read_json.py` is rerun, run `p1pb_integrate.py --json-only` after it.

## Files (hashes in `P1PB_FILES.sha256`; each exists in this one place only)
`SPEC_P1PB.md`, `SPEC_P1PB.sha256`, `P1PB_ADMISSIBILITY.json`, `P1PB_HOLDOUT_60.json`, `p1pb_admissibility.py`, `p1pb_forward.py`, `p1pb_read.py`, `p1pb_finalize.py`, `p1pb_integrate.py`, `P1PB_TICK0_s{0,1}.npz`, `P1PB_FW_CONTROL.json`, `P1PB_F1_CONTROL.json`, `P1PB_PERWINDOW.npz`, `P1PB_RESULT.json`, `P1PB_RESULT.md`, three logs, scratch probes in `p1pb/`. Rerun: `p1pb_admissibility.py`, `p1pb_forward.py control`, `p1pb_forward.py holdout`, `p1pb_read.py`, `p1pb_finalize.py`.
