# Post-training zero-gradient gate (`stack/scripts/zerograd_gate.py`) -- 2026-10-18

Deliverable: the durable instrument that CLAUDE.md ("BUILT, TESTED, AND UNREACHABLE FROM ITS CALLER") said
"belongs in `stack/scripts/` as a POST-TRAINING GATE". Origin: `incoming/2026-09-10-refcv5v2-zerograd-heads/probe_zerograd3.py`
(its idea -- Adam state is allocated lazily, so a registered parameter with no state never got a gradient -- is kept).

## What it checks
| case | meaning | catches |
|---|---|---|
| (a) `NO_STATE` | no optimizer-state entry: `.grad` was None every step | head detached / never called / not in the loss graph |
| (b) `ZERO_STATE` | entry exists, every float state tensor (`exp_avg`, `exp_avg_sq`) EXACTLY 0 | `--w-x` with `default=0.0`: `0.0 * term` is a zeros grad tensor, so Adam allocates state and a count of "registered vs with state" reads clean |

Exit 0 PASS, 1 FAIL (untrained parameter or a failed `--control`), 2 ERROR (un-evaluable checkpoint is never a PASS).
Naming is an exact count of all order-preserving shape alignments (not the probe's greedy match); an ambiguous
row lists every candidate and `--allow` exempts it only if every candidate is allowed. Verdict never depends on naming.

## Acceptance on real data -- MEASURED
Input: `D:/Projects/TanitAD-artifacts/refcv5v2_final/ckpt.pt`, 1,299,559,201 bytes, sha256
`a42d68af682521cd8f2f8ca2c4d5949a34e352ce1074351ebb16cc16e9dce57d`, loaded `weights_only=True` + mmap, CPU only.
Artifact: `raw/zerograd_refcv5v2_final.json` (+ `.txt`); allow-arm `raw/zerograd_refcv5v2_final_allowed.json`.

* MEASURED: 357 registered, 351 with state, 351 trained, case (a) = **6**, case (b) = **0**, optimizer step max 40,284.
* MEASURED -- **reproduces the banked finding exactly**, three heads, **18,472** parameters, all case (a):

  | head | tensors | numel (gate) | banked (INHERITED, CLAUDE.md) |
  |---|---|---|---|
  | `core.decoder.offset_head` | `.weight` 6,144 + `.bias` 16 | **6,160** | 6,160 |
  | `tac_goal_tok_head.net` | `.weight` 11,264 + `.bias` 22 | **11,286** | 11,286 |
  | `scorer.goal_point` | `.weight` 1,024 + `.bias` 2 | **1,026** | 1,026 |

  Verdict **FAIL**, rc 1. Not tuned to match: the gate was run before any naming refinement and gave the same six ids.
* MEASURED naming: the alignment is **unique** (n_alignments = 1) over 361 float `state_dict` entries after excluding the
  132 BatchNorm running statistics; 0 of 357 names ambiguous. The 4 non-parameter float entries left over are
  `core.lat_log_prior`, `core.lon_log_prior`, `core.decoder.anchors`, `core.decoder.anchor_controls`
  (consistent with the probe's "over-counts by 3" plus its hard-coded `.anchors`).
  Without the BN exclusion 132 of 357 ids are name-ambiguous (a BN `weight`/`bias` next to same-shaped `running_*`) --
  the verdict is identical; only the names blur. That is why the exclusion is a documented default.
* MEASURED controls (same breath, must read trained): `str_goal_head.*` 2, `scorer.[^g].*` 2, `core.encoder.*` 198,
  `lat_head_tac.*` 2 trained matches -- the gate can see trained heads, so its 6 are not a blind probe.
* MEASURED allow-arm: `--allow` on exactly those three heads -> rc 0, allowed numel 18,472, the deliberately wrong
  extra pattern is reported `unused`.
* MEASURED wall time (this box, external D: drive): 2.4 s warm cache without `--sha256`; 3.4 s warm with it; 34.8 s on the very first (cold-cache, with `--sha256`) run.

## Differences from the probe -- stated, not hidden
* Case (b) is new (probe: case (a) only). It is zero on refcv5-v2, so it did not change this result.
* The probe's control loop printed nothing for a control whose name did not exist (`nav_inject`/`ego_inject` have no entry
  in this model's `state_dict`); the gate FAILS a `--control` that matches no trained parameter.
* UNVERIFIED observation, not a gate finding: `lat_head_tac`/`lon_head_tac` (4 tensors) have Adam step 40,134 vs 40,284 for
  every other trained parameter (0.9963) -- they got no gradient in 150 steps. Cause not investigated (`info.partial_grad`).

## Tests -- MEASURED
`stack/tests/test_zerograd_gate.py`, run from a temp tree of `stack/` exported from tip `3a28bc1977c7e98599474299365e5f41d9c653ee`:
**14 passed in 5.21 s** (`python -m pytest tests/test_zerograd_gate.py -q`). Real Adam steps; control PASS; case (a) and (b) FAIL
naming the head; `--allow` exact (fullmatch); two MUTATION arms (disable (b) -> the zero-weight head PASSES; disable (a) -> the
detached head PASSES); a real source mutation (`ZERO_STATE -> TRAINED`) turned 5 tests red; DP checked against brute force;
ambiguity fails closed; weights-only / SGD / missing file -> rc 2; subprocess CLI contract + sha256.

## Known blind spots (also in the module docstring)
Frozen-by-design parameters look identical to broken ones (use `--allow`); coupled-L2 Adam hides case (b); conditional heads are
only listed under `info.partial_grad`; plain SGD is rc 2.

## Integration (Master Mind's -- not done here)
Nothing calls this yet. Candidates: `stack/ops/sup_refcv7.sh` before `write_done_marker` (rc != 0 => no done-marker/publish),
and the end of `refc_v3_train.py` runs; adding `"param_names": [n for n,_ in model.named_parameters()]` to the checkpoint payload makes
naming exact (that payload is test-pinned: `tests/test_refc_v3_save_before_eval.py`).
