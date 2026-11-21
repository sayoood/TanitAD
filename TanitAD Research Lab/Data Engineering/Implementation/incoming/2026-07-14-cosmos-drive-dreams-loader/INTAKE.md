# INTAKE — Cosmos-Drive-Dreams loader (publicly-claimable synthetic AV corpus)

- **Package:** `Data Engineering/Implementation/incoming/2026-07-14-cosmos-drive-dreams-loader/`
- **Author agent / date:** Data Engineering agent (Tuesday), 2026-07-14
- **Proposed target:** `stack/tanitad/data/cosmos_drive.py` (+ `stack/tests/test_cosmos_drive.py`)
- **Hypothesis / WP served:** D-014 sim arm · H7 (long-tail scene diversity) · H4 arm-B · D-010 mix

## What & why (≤10 lines)

The 2026-07-07 license review (D-002) excluded the **real** PhysicalAI-AV sets from every public
claim (internal-dev-only / confidential / 12-mo). D-014 then named the two ungated NVIDIA synthetic
corpora as the training-mix sim arm, of which **Cosmos-Drive-Dreams is CC-BY-4.0 — the one AV asset
we may render, train on, and cite publicly.** This is its first loader. It turns each synthetic clip
(`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`, RDS-HQ format) into the *same* episode
contract as comma2k19 / physicalai: derives (steer, accel) + (x, y, yaw, v) from the per-frame 4×4
`vehicle_pose` sequence (no CAN here), resamples 30 Hz→10 Hz, D-015 3-frame/9-channel stacks, D-016
focal canonicalization (front_wide_120fov = the same 120° HFOV as PhysicalAI → identical nominal
focal). `CORPUS_META` is byte-identical to comma2k19 (D-017 I7) so probes and `MixedWindowDataset`
(D-010) admit it into the real+sim mix. Long-tail weather/night/intersection scenes are exactly the
distribution comma2k19's highway commute lacks. Research note:
`Data Engineering/Research/2026-07-14-cosmos-drive-dreams-loader-and-landscape.md`.

## Evidence & tests

- Tests: `tests/test_cosmos_drive.py` — **9 passed / 4.0 s** on the venv (`C:\Users\Admin\venvs\tanitad`).
  Full stack suite unaffected: **73 passed, 1 skipped**.
- Covered: signal derivation on an analytic constant-speed/constant-yaw arc (v, accel≈0,
  steer=atan(L·κ)); low-speed 1/v steer guard; D-015 9-channel contract (`assert_contract(channels=9)`);
  **I7 fingerprint identity with comma2k19** (`CORPUS_META == comma2k19.CORPUS_META`, `LICENSE == "CC-BY-4.0"`);
  clip-level (I3) split disjointness; per-weather episode-id distinctness; filename/weather parsing;
  video↔pose pairing discovery + `[N,4,4]` pose IO; **mix-admissibility** (`MixedWindowDataset` accepts it).
- Zero real bytes / zero `av` in CI (decode + pose IO injected). Real-clip validation is a documented
  pod step: `verify_real_clip()` returns A8 + speed/steer/accel ranges for the data card (P8 — not yet run).

## Risk & rollback

- Blast radius: **additive only** — one new module + one new test file; no existing file touched.
  0 new dependencies (`av` already in `.[real]`; numpy/torch/pandas present).
- Known limitations (P8, in the module docstring): the exact `vehicle_pose` filename glob and the
  FLU/OpenCV axis order are taken from RDS-HQ toolkit docs and **pod-verified** via `verify_real_clip`
  before any trained claim; synthetic pixels ≠ off-expert action-consequence rollouts (that stays with
  CARLA-on-pod). Neither blocks integration — the contract/signal code is proven on fixtures.
- Rollback: delete the two files; nothing imports them yet.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate
- **Date / by:** 2026-07-08 (overnight), MVP orchestrator (autonomous loop iteration 4)
- **Reason & notes:** Strategically important package: the ONE corpus we can train on AND cite
  publicly (CC-BY-4.0), contract- and fingerprint-identical to comma2k19 (I7 test), analytic signal
  derivation validated, honest pod-verification step documented before any trained claim
  (`verify_real_clip`). Only change: test import de-hacked. Full suite 128 passed / 1 sim-skip.
  Follow-ups queued: pod `verify_real_clip` run + data card; then cosmos joins the D-010 mix
  (bake-off-gated share).
- **Integrated as:** `stack/tanitad/data/cosmos_drive.py` + `stack/tests/test_cosmos_drive.py`
  (see `intake(data-eng)` commit, 2026-07-08)
