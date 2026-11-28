# SFTT launch gate, G-LIVE L5: read NAVSIM's PDMResults, compare at the label's stored dtype (MM 2026-11-23)

Built on commit 95b3d174, the L2 census + field_reach amendment. Two files change: `refe/gate_checks.py` and `refe/test_scorerft_profile.py`. The patch is `patch_l5b.py`, applied to 95b3d174's two files.

## What failed (MEASURED)
**SFTT gate 1** (package B, 13:45Z) failed L5 with *"NAVSIM is not importable here"*:
- The pod's vendored DriveRL nuplan-devkit lacked `planner/ml_planner`.
- The MM installed `ml_planner/{__init__.py, transform_utils.py}` from the dev-box NAVSIM harness copy. Its 7 shared dependency modules are byte-identical to the pod's once CR is stripped.

**SFTT gate 2** (package C = 95b3d174, 15:49Z) passed L2 (the gate head through the census) and failed L5:
*"the identity probe raised AttributeError: 'PDMResults' object has no attribute 'iloc'"*.
- navsim 3e8291b's `pdm_score` returns `-> PDMResults` (`navsim/evaluate/pdm_score.py:89`). PDMResults is a dataclass with `no_at_fault_collisions … driving_direction_compliance, score` (`navsim/common/dataclasses.py:557`). It is not a DataFrame.
- **L5 had never run anywhere before today.** That is the "built, tested, unreachable" class: the probe was written against another NAVSIM version's return type.

## The pod probe (l5probe, 15:55Z)
Setup:
- package C copied to `/workspace/data/refe_p5c/l5probe/`, with only the `.iloc` read fixed;
- `launch_gate.py run --profile refe_scorerft_t --only G-LIVE`, with the real SFTT argv;
- banked here as `l5probe_evidence_G-LIVE.json`, `l5probe.log` and `l5probe_run.sh`, pulled line-framed and sha-checked.

Results:
- **L5 compared 8 tokens × 6 sub-scores. The ONLY difference:** `ep` NAVSIM `0.9281340780250414` (float64) vs label `0.9281340837478638`. That label is exactly `float32(NAVSIM)`.
  - The labeller stores float32, so a float64 identity at tol 1e-9 can never hold.
  - Max |diff| 5.72e-9 is the float32 rounding of one value.
- **L2 PASSED:** ep from the smoke, gate from the registered census.
- The probe's other "FAIL" lines are expected for a scratch probe: a modified tree, and checks skipped by `--only`.

## The fix
- **`_pdm_subscores(res, names)`** reads the six sub-scores from a PDMResults dataclass (by attribute) or a one-row DataFrame (by column).
- **`identity_diff(navsim, label, label_dtype)`** clips NAVSIM's value to [0, 1] and casts it to the LABEL ARRAY's dtype before the difference.
  - The tolerance is unchanged at 1e-9, so for float32 labels the identity is now EXACT at the stored precision.
  - float64 labels are still compared at float64.
- The facts also record `compared_at_dtype`, `max_abs_diff_float64` (the raw number, never hidden) and up to 24 `diffs`.

## Tests
New check `l5_identity_dtype` in `test_scorerft_profile.py`, built on the pod's measured numbers:
- dataclass and frame reads agree;
- the label is float32(NAVSIM);
- a float32 label equal to float32(NAVSIM) gives PASS;
- a float32 label ONE ulp away gives FAIL;
- float64 labels at 5.7e-9 give FAIL;
- clipping is applied.

Two mutations go RED: comparing at float64 regardless of dtype (the gate-2 defect), and always using `.iloc`. The 13-run registration set is in `tests.tsv`.

## What it does NOT change
- L1 to L4, the L2 census and field_reach are unchanged.
- The card and DESIGN are unchanged.
- The trainer, labeller and rank_loss are byte-identical to package A3 (C's trainer).
