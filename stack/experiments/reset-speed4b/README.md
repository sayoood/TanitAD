# Speed/scale reset + REF-A 4-brain — artifacts (2026-07-14 → 15)

Durability copy of the reset code that ran on the pods (`/workspace/tmp`). These
are the exact working scripts behind the 2026-07-15 program report. Paths inside
them are pod-absolute (`/workspace/...`, `sys.path` inserts) — they run **on the
pods**, not from this checkout; this dir is an archive + record.

## The bug and the fix
Actions are derivatives `[steering, acceleration]`. To produce an absolute metric
displacement the model must know the **current speed** `v0`. None of the arms fed
it — they inferred speed from vision, which a frozen DINO encoder cannot
(speed-probe R² ceilinged at 0.61). Fix: append `v0 = pose_last[:,3] / 10`
(last observed frame — leakage-safe) as a **3rd action channel** (`action_dim` 2→3).
Validated in isolation: REF-A operative **3.73 → 0.83 m** in-training, speed R²
**0.61 → 0.965**. All three arms were restarted from scratch with speed + jerk +
aux-accel; flagship and REF-A also carry the full 4 brains.

## Files
| File | What | Pod origin |
|---|---|---|
| `refa_train_plus.py` | REF-A 4-brain trainer — `--four-brain` builds `RefAModelPlus.from_stack_config(flagship4b_config())`; `FeatureWindowDataset4B` derives maneuver + genuine 25 s nav labels; `compute_losses_plus` adds strategic-route / tactical-maneuver / goal-waypoint / goal-latent / tactical-predictor losses (same objective as `flagship_losses.flagship_loss`). Frozen-DINO adapter is the only diff from the flagship. | pod3 `/workspace/tmp/refa_plus/` |
| `refa_plus.py` | `RefAModelPlus` (temporal adapter + 4-brain via `**kw`) + `build_aux_heads`. | pod3 `/workspace/tmp/refa_plus/` |
| `eval_refa4b_grounded.py` | REF-A 4-brain grounded-rollout gate (`--four-brain --speed-input --adapter temporal`); appends v0 at eval. | pod3 `/workspace/tmp/refa_eval/` |
| `eval_grounded_rollout_4b_speed.py` | Flagship grounded-rollout gate, speed-input aware (`--config flagship4b --speed-input`). | pod2 `stack/scripts/` |
| `render_arm.py` | Unified BEV overlay renderer (`--arm refa|flagship|refb`) + coords dump. | local |
| `build_svg_gallery.py` / `build_ascii.py` | Lightweight vector gallery + ASCII trajectory plots (mobile-safe delivery). | local |
| `*_gate_*.json` | Decision-grade gate results (held-out, 40 eps, 8 splits). | pods |

## Run commands (on the pods)
```
# REF-A 4-brain (pod3)
python3 refa_train_plus.py --data-root /root/phase0_dinofeats \
  --out /workspace/experiments/refa-4brain-speed-30k --four-brain --speed-input \
  --aux-egomotion --adapter temporal --aux-accel --jerk-weight 0.02 --rollout-k 12 --steps 30000

# REF-A 4-brain gate (pod3)
python3 eval_refa4b_grounded.py --ckpt .../refa-4brain-speed-30k/ckpt.pt \
  --cache-dirs /root/phase0_dinofeats --four-brain --speed-input --adapter temporal \
  --episodes 40 --batch 8 --stride 8 --n-splits 8 --out .../refa4b_gate_30k.json

# Flagship gate (pod2)
python3 eval_grounded_rollout_4b_speed.py --ckpt .../flagship4b-speedjerk-30k/ckpt.pt \
  --cache-dirs /workspace/data/physicalai_phase0/_epcache --config flagship4b --speed-input \
  --episodes 40 --batch 8 --stride 8 --n-splits 8 --out .../flagship_5k_gate.json
```

## Decision-grade results (held-out, vs CV = 0.83 m)
- **REF-A 4-brain 30k FINAL: 2.14 m** — halved old REF-A (3.73 m) via speed+4-brain, but
  **plateaued at the frozen-DINO ceiling** (14k 2.05 → 30k 2.14), above CV. Arm DONE.
- **Flagship 5k: 2.34 m** overall, but **better @1s than REF-A@30k on every stratum**
  (gentle 0.63 / sharp 0.98 / straight 1.01) — trained encoder sharper near-term at ⅐ the
  training; 2 s endpoint unstable at 5k (de@2s 4.60). Fair verdict = flagship@30k.
- **Neither beats CV yet** — highway constant-velocity is near-unbeatable on straights.
