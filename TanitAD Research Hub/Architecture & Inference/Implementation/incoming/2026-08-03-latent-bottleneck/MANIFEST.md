# D-LATENT — deliverable manifest

**Stream** latent / world-model representation · **Date** 2026-08-03 · **0 pod GPU-h**, no training
pod touched (`tanitad-new` / `tanitad-pod4` untouched; Thor untouched). All compute on the dev box
RTX 4060 with `OMP_NUM_THREADS` set (the documented multi-arm trap).

## Artifacts and where they live

| artifact | repo path | in ONE place only? |
|---|---|---|
| **Main deliverable** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-latent-bottleneck/LATENT_BOTTLENECK.md` | no — staged in git |
| **Pre-registration** (both outcomes fixed in advance) | `Project Steering/PREREG_TEMPORAL_LATENT.md` | no — staged in git |
| ⭐ mechanism results | `…/2026-08-03-latent-bottleneck/results_mechanism.json` | no — staged |
| precision-ladder results | `…/2026-08-03-latent-bottleneck/results_precision_ladder.json` | no — staged |
| D-probe results (pass 2, corrected substrate) | `…/2026-08-03-latent-bottleneck/results_temporal_falsifier.json` | no — staged |
| D-probe pass 1 (defective substrate, KEPT as evidence) | `…/2026-08-03-latent-bottleneck/raw/results_pass1_STACKAVG_INADMISSIBLE.json` | no — staged |
| approach-A cost measurement | `…/2026-08-03-latent-bottleneck/raw/temporal_kv_cost.json` | no — staged |
| runners | `…/2026-08-03-latent-bottleneck/run_mechanism.py`, `run_precision_ladder.py`, `run_temporal_falsifier.py`, `analyze_temporal_kv_cost.py` | no — staged |
| logs | `…/2026-08-03-latent-bottleneck/raw/run_log*.txt` | no — staged |
| **Instrument change** (reusable) | `stack/tanitad/eval/accel_probe.py` — added the `tdiff` / `abstdiff` adjacent-frame feature bases | no — staged |
| **Instrument test** | `stack/tests/test_accel_probe.py` — `test_adjacent_frame_bases` | no — staged |

### ⚠️ Lives in ONE place (dev box only) — derived caches, rebuildable from the staged runners

| file | size | rebuild |
|---|---|---|
| `C:/Users/Admin/tanitad-data/eval/dlatent_pixel_substrate.pt` | ~1.1 GB | `python run_temporal_falsifier.py --stage substrate` (≈95 s) |
| `C:/Users/Admin/tanitad-data/eval/dlatent_pixel_substrate_pass1.pt` | ~0.9 GB | pass-1 variant, kept for the documented defect; rebuildable by reverting `latest_frame` |

**Not staged and deliberately so:** the two `.pt` caches are derived data, deterministic from the
banked comma2k19 episode cache + the staged code, and too large for the repo. Neither is an input
that took real effort to produce (95 s each). The banked inputs they derive from —
`C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f/ep_*.pt` and
`…/idm_derived_accel_latents.pt` — pre-date this stream and are owned by the IDM stream.

## Reproduction

```bash
cd "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-latent-bottleneck"
OMP_NUM_THREADS=4 python run_mechanism.py        --n-boot 2000 --out results_mechanism.json
OMP_NUM_THREADS=4 python run_precision_ladder.py --n-boot 2000 --out results_precision_ladder.json
OMP_NUM_THREADS=6 python run_temporal_falsifier.py --stage all --n-boot 2000 \
    --out results_temporal_falsifier.json
OMP_NUM_THREADS=2 python analyze_temporal_kv_cost.py --out raw/temporal_kv_cost.json
```

## Escalations — these need someone else to act (repeated from §8, because a README is not a channel)

1. **`Project Steering/GATE_PROTOCOL.md`**: adopt the §6 latent screen as a pre-flight gate for any
   encoder-training authorisation. `dynenc-branchB` spent 40 k steps on a latent this screen would
   have rejected in minutes.
2. **`Project Steering/BACKLOG.md` B5** (frozen V-JEPA 2 video-pretrained encoder): top-ranked
   encoder experiment by this analysis; re-scope it to run the screen FIRST.
3. **`stack/scripts/idm_head.py` docstring** and
   `…/2026-08-03-idm-accel-recoverability/ACCEL_RECOVERABILITY.md` §4.7: the *"σ ≲ 0.1 m/s, ~47×"*
   precision requirement is estimator-specific (2-point centred difference). With the optimal
   9-point Savitzky-Golay derivative it is **σ ≲ 0.28 m/s, ~21×**.
4. **`Project Steering/BACKLOG.md` A7** (Delta-JEPA): second independent refutation — the true
   adjacent-frame difference bases are at the null and the direct-increment regression gives
   corr +0.0007.
5. **`Project Steering/RETRACTION_LOG.md`**: the "single RGB frame" framing that this stream and the
   sitclf-temporal stream were both briefed with is factually wrong about our input
   (`in_channels=9`, D-015 3-frame sliding stack). Root-cause class: *an architectural claim
   inherited from a code READING rather than from a shape MEASUREMENT.*

## What is NOT done

* The `mot*` / `stk*` / rbf pixel arms answer the pre-registered L-vs-V question only if their
  positive control fires; if it does not, that family stays VOID and the L-vs-V question remains
  **OPEN**. See `LATENT_BOTTLENECK.md` §4.4 for the measured outcome.
* The §6 screen has **not** been run on v5f, on REF-C's ResNet trunk, or on PhysicalAI-AV. Those are
  the arms the programme is actually deciding about.
* No retrain was launched and none is proposed here without the screen first.
