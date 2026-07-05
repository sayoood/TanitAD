# STATE — Data Engineering

LAST_RUN: 2026-07-07 (W1→W2, Tuesday agent)
QUALITY: full (G-A…G-F, G-D1, G-D2 met; suite 40✓/1 skip)

## This run (2026-07-07)
- Entry: backlog #1 (comma2k19 loader) found **already shipped by Sayed as D-009** mid-session
  (`comma2k19.py`, `base250cam` 6-ch 2-frame contract). Did NOT overwrite; pivoted to validate + close gaps.
- **Real-data validation** (HF token via `tanitad.keys`): loader's `_decode_video` decodes real comma HEVC
  → [200,3,256,256] @ ~105 fps; A8 real ≈ 0.053. Decode path de-risked before A40 spend.
- **Contract reconciliation:** generalized shared `assert_contract(ep, channels=1|6|None)`; added
  `test_comma2k19_contract.py` (ties D-009 loader to the shared contract home + A8 check); exported
  comma2k19 from `tanitad.data.__init__`.
- **Docs:** data card (`2026-07-07-comma2k19-data-card.md`, all G-D1 fields) + research note
  (`2026-07-07-comma2k19-validation-and-h7.md`, H7 deltas + findings).
- **Findings:** Windows `|` path bug (extract on Linux pod only); A8 weak on raw highway RGB → change-
  weighting justified; H7 gains LAOF + Sensorimotor-WM.

- [x] **Per-dataset A8 statistics harness** (backlog #4) — `stack/tanitad/data/stats.py` shipped (6 tests);
  measured toy(0.046, threshold-insensitive) vs comma-real(0.053→0.012, threshold-sensitive); feeds
  change-weighting + D-010 mix. Finding in the research note §4a.

- [x] **PhysicalAI-AV license review** (backlog #2, D-002) — real AV sets = NVIDIA internal-dev-only,
  confidential, 12-mo → **no public claims**; Cosmos-Drive-Dreams (CC-BY-4.0) is the publicly-safe AV
  asset; comma2k19 (MIT) stays the public corpus. Note: `2026-07-07-physicalai-av-license-review.md`.

## Next (backlog, priority order)
1. **Steering-ratio calibration log** (H7 artifact) — per-segment residual when seed IDM trains on comma2k19.
2. If a publicly-claimable *synthetic* AV corpus is wanted, harden a **Cosmos-Drive-Dreams** loader
   (CC-BY-4.0) — NOT the gated real PhysicalAI-AV sets (former backlog #2 target, now legally excluded from
   public claims).
2. **On the Linux A40 pod:** pull+unzip Chunk_1 (8.7 GB via `scripts/extract_comma2k19.py`), add `av` to
   `[real]` extra, smoke `build_episode` on 3 segments, wire into Stage-A. ~1–2 engineer-h, 0 new code.
3. **Run the A8 harness on real Chunk_1** (many segments) once on the pod → set the change-weight schedule
   and the D-010 real-vs-sim consequence comparison from real distributions, not one clip.
4. PhysicalAI-AV loader hardening + license review note (backlog #2, D-002) — second corpus.
5. Focal-canonicalization prototype (backlog #3) — deprioritized until heterogeneous video (GoPro/BDD/OpenDV).

## HANDOFF
None — all work committed and pushed, suite green. Real segment cached in scratchpad (not committed).
