# PLAN — W6 (E5 resumed): nuScenes open-loop planning harness + input adapter

**Briefs:** `BRIEF.md` (E5) + `…/2026-09-19-eval-suite-build/BUILD_PLAN.md` row W6 and §1 contract.
**Owns (exclusive write):** `taniteval/adapters/nuscenes_planning.py`,
`taniteval/tests/test_nuscenes_planning.py`, this folder. Everything else is read-only; changes to
other owners' files ship as PATCH files here (W2: criteria guard; W4: published rows) or as asks in
`COMMS.md` (W1: register `nuscenes_ol`).

## Priority order (a priority order, not a dependency chain)

1. **Harness core** — kernels verbatim to the pinned reference code (ST-P3, UniAD, VAD; AD-MLP reuses
   ST-P3's), both reductions (`uniad-noavg` at-t / `stp3-temavg` averaged-up-to-t), a tagged-result
   API that cannot emit an untagged or mixed number, `claim_bearing:false` with no way to set it true.
2. **Tests** — the SPEC.md literals (analytic L2 targets, PARA-Drive real-data cross-check, literal
   collision cells per kernel, sample-set counts, fillPoly port masks), the source-level RED mutation,
   API-impossibility tests. Dataset-free, CPU only.
3. **nuScenes metadata → harness inputs** — GT (LiDAR-origin, all three pipelines), per-pipeline
   sample sets / masks, occupancy builders (UniAD, VAD; ST-P3's warp-based builder is a named gap),
   the `cv2.fillPoly` port, trivial arms (GT control, STOP, CV).
4. **Input adapter** — CAM_FRONT pinhole → the cylindrical training frame via `calib.pinhole_rectify`
   (never a resize), observed mask stamped; history slot construction (ST / NT / EXACT) and the data
   tier each needs; plan (our ego frame) → LiDAR frame via the lever arm.
5. **§1 run directory** — `bench_run.json`, `scores/<arm>.csv`, `artifacts/<arm>.json`,
   `criteria/<arm>.txt`, `summary.json`, so W1 can register `nuscenes_ol` with one import.
6. **Criteria guard PATCH for W2** — EXTERNAL_ONLY scope + a violation when a nuScenes planning number
   is used as a TanitAD criterion; its test; `git apply --check` and a scratch-mirror test run.
7. **External rows** — every number re-read from a banked PDF (library key + table + page) →
   `raw/nuscenes_external_rows.md` (E4/W4) and `code/published_results_nuscenes.patch.json` (W4).
8. **PI ACTION LIST** — registration + ToU steps, minimal file set with bucket LIST/HEAD sizes, the
   D: landing layout (exFAT 1 MiB clusters), a fetch script that REFUSES to run without the PI's
   acknowledgement (never run by an agent).
9. `RESULT.md`, `COMMS.md`, stage exact paths, blob-verify at the end, manifest.

## Constraints honoured

No nuScenes bytes (bucket LIST/HEAD only); no registration/terms; CPU only (GPU: A8 then A7 — never
touched); no sub-agents; stage exact paths, never commit/push; evidence class on every claim.
