# Ruling — the Thor NavSim CPU backend (Master Mind, 2026-10-05 ~00:30 Europe/Berlin)

## 1. The registered bar FAILED as written
DESIGN.md's validation bar was max abs diff 0.0 on every column. MEASURED results (`raw/THOR_BACKEND_VALIDATION.json`):
* **navtest R7_A1 @ 30,000:** 34 / 12,146 tokens differ, max 1.2e-11. **navhard R7_A1 @ 50,400:** 54 / 5,912 differ,
  max 1.0e-11.
* No discrete flips, and the split means differ by ≤ 1.1e-15. EPDMS reads 20.2174 and the navtest average 71.8849 on both
  machines.
* Cause (MEASURED, `raw/libm_probe_summary.json`): the C math library. Windows CRT on x86-64 and glibc 2.39 on aarch64
  differ by ≥ 1 ulp on a few % of `exp` / `cos` / `sin` / `arctan2` values. Package versions, the devkit and every data tree
  are identical.

## 2. Policy adopted (set AFTER seeing the data; disclosed as such)
This is an instrument-equivalence policy, not a scientific bar. Its tolerances sit 9+ orders of magnitude below every
decision bar in the programme (e.g. ΔEPDMS ≥ 0.020).
1. **Cross-backend admissible** when there are 0 discrete flips (a sub-score multiplier, or a token's valid / PASS
   state), per-token |Δ| ≤ 1e-9, and split-mean |Δ| ≤ 1e-12. Every Thor artifact carries `backend: thor`.
2. **Backend-consistent by default for NEW comparisons.** All arms of one comparison are scored on the same backend,
   including the reference / floor arms. A comparison that mixes backends states it, with the tolerance above.
3. **The AgentInput fingerprint fallback is ACCEPTED** (`code/tanitad_seam_agent_ulp.py`). It applies only when a
   dev-box reference re-hashes to the seam fingerprint and Thor's values are within 1e-9 of it. Disclosure: atol 1e-9 was
   chosen after observing max 3.6e-15. 481 / 5,912 navhard tokens (all synthetic stage-2) need it.

## 3. Routing
* The refcv7 step-50,400 milestone stays on the dev box: it is partly scored there, and its governed queue owns the
  claims. Thor scores the same open arms into `…/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/thor_scores/`
  as early reads and a cross-backend check, never under the package's standard names.
* NEW NavSim work goes to Thor first: the refcv7 LEGAL row's scoring (R7_VMAXOFF) once its seams exist, D6 P4's
  changed-pick scoring, and every refcv8 NavSim row.
* Thor safety: ≤ 4 scorers, launch only at MemAvailable ≥ 35 GB, SIGTERM below 20 GB, and the `NAVSIM_YIELD` file is
  honoured.
