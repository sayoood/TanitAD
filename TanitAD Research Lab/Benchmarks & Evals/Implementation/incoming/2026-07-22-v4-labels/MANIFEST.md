# v4 training label set — mint + wiring + proof (2026-07-22)

The pre-launch blocker closed: v4's two marquee heads (factorised LAT×LON×DIST
tactical, strategic goal scalars) trained on **IGNORE_INDEX** because the labels
were never minted and the dataset never emitted them. This mints the full v4 label
set on the parity corpus, wires the dataset, and proves the heads train on real
(non-ignored) targets.

## Deliverables (repo — STAGED)

| Artifact | Path | What |
|---|---|---|
| Label minter (pure fns + corpus builder CLI) | `stack/scripts/v4_labels.py` | mints lat/lon/dist tactical targets, v3 route token, strategic scalars (ttm·curv@3s·curv@5s·tspeed@5s); `build` writes a v15-format cache + provenance + parity check |
| Dataset wiring | `stack/scripts/flagship_v4_data.py` | `FlagshipV4Dataset(FlagshipWindowDataset)` — additive, emits exactly the keys `train_flagship_v4.v4_loss_step` reads; v1/v2.1 keys byte-identical |
| Strategic-scalar loss | `stack/tanitad/train/v4_curriculum.py` | `strategic_scalar_loss` — masked, per-channel-scaled smooth-L1 (§4.3/§7A.4) |
| Goal-scalar head | `stack/tanitad/models/strategic_goal.py` | `GoalScalarHead` (§3.1 geometry; P6 consumer stand-in) |
| Trainer wiring | `stack/scripts/train_flagship_v4.py` | `v4_loss_step` now computes the strategic term when a goal head is passed (byte-identical when not); `--real-smoke` runs the proof on real frames |
| Tests | `stack/tests/test_v4_labels.py` | 14 tests: vocab widths, masking, kinematics, both losses, dataset additivity, head size |
| Provenance (train) | `incoming/.../labels_train_v4_provenance.json` | corpus coverage + parity proof (see below) |
| Provenance (val) | `incoming/.../labels_val_v4_provenance.json` | val coverage + parity |
| Proof A | `incoming/.../proof_A_heads_train_on_real_targets.json` | factorised CE + strategic-scalar loss non-zero, grads into all heads |

## Deliverables (pod-side — multi-GB tensors stay on the pod)

| Artifact | Location |
|---|---|
| Train label cache | `tanitad-pod2:/workspace/v15/labels_train_v4.pt` |
| Val label cache | `tanitad-pod2:/workspace/v15/labels_val_v4.pt` |

The *recipe* (`v4_labels.py`) is in the repo; the multi-GB label tensors are the
cache and stay pod-side, indexed by the same `eids` the v1.6 loader uses.

## Parity

Labels are re-derived on the existing parity pose cache
(`/workspace/v15/poses_train.pt`, 2376 eps) — nothing re-selects/reorders/drops an
episode. Window count and the v2.1 fields are verified against the shipped v1.6
cache (see provenance `v21_parity`).

## Honest mintability (what v4 can and cannot mint) — see `v4_labels.mintability_report()`

- **Fully minted (kinematic):** LATMANEUVER (7), LONMODE (6, ego-only), DIST band
  to next route maneuver, route (v2.1 class + v3 token where confirmable),
  route_graded, vt_band, ttm, curv@3s, curv@5s, tspeed@5s.
- **NOT mintable → IGNORE / never emitted:** LONMODE `follow_lead/close_gap/open_gap`
  (need lead state — a `None` stub), LATMANEUVER `merge_in/yield_merge` (need
  another agent), ROUTE `straight`/robust `exit`/`merge`/`roundabout` (need a map),
  TACPOINT name (position minted, name needs vision/map). Each reaches its head as
  IGNORE with its coverage stated, never faked.
